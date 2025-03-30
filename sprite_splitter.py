import os
import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing

def remove_white_border(image, white_threshold=220):
    data = np.array(image)
    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    near_white = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)
    faint = a < 255
    data[near_white & faint] = [0, 0, 0, 0]
    return Image.fromarray(data)

def boxes_overlap(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)

def merge_boxes(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))

def box_area(box):
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0)

def is_contained(box_small, box_big):
    """
    Returns True if box_small is fully inside box_big.
    box = (x0, y0, x1, y1)
    """
    x0s, y0s, x1s, y1s = box_small
    x0b, y0b, x1b, y1b = box_big
    return (x0b <= x0s) and (y0b <= y0s) and (x1b >= x1s) and (y1b >= y1s)

def filter_contained_boxes(boxes):
    """
    Removes any box that is fully contained in a bigger box.
    We only discard if the containing box's area >= contained box's area.
    """
    final = []
    for i, box in enumerate(boxes):
        contained = False
        for j, other in enumerate(boxes):
            if i == j:
                continue
            if is_contained(box, other) and box_area(other) >= box_area(box):
                contained = True
                break
        if not contained:
            final.append(box)
    return final

def boxes_close_in_2d(boxA, boxB, dist=10):
    """
    Returns True if the minimal gap between boxA and boxB in both x and y
    is <= dist. This allows merging of boxes that are near each other
    (horizontally or vertically) by up to 'dist' pixels.
    """
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Horizontal gap
    if xA2 < xB1:
        gap_x = xB1 - xA2
    elif xB2 < xA1:
        gap_x = xA1 - xB2
    else:
        gap_x = 0

    # Vertical gap
    if yA2 < yB1:
        gap_y = yB1 - yA2
    elif yB2 < yA1:
        gap_y = yA1 - yB2
    else:
        gap_y = 0

    return (gap_x <= dist) and (gap_y <= dist)

def merge_all_close_boxes(boxes, dist=10):
    """
    Iteratively merges bounding boxes if they overlap or are close in 2D
    until no more merges occur.
    """
    boxes = boxes[:]
    changed = True
    while changed:
        changed = False
        new_boxes = []
        for box in boxes:
            merged = False
            for i, existing in enumerate(new_boxes):
                if boxes_overlap(box, existing) or boxes_close_in_2d(box, existing, dist):
                    new_boxes[i] = merge_boxes(existing, box)
                    merged = True
                    changed = True
                    break
            if not merged:
                new_boxes.append(box)
        boxes = new_boxes
    return boxes

def reorder_boxes_grid(merged_boxes, expected):
    """
    Reorder the list of boxes so that they are arranged in grid order.
    We assume the grid is roughly square. For example, if expected==16, we try to produce 4 rows.
    """
    if not merged_boxes:
        return merged_boxes

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in merged_boxes]
    grid_rows = int(round(expected ** 0.5)) or 1

    # Sort by vertical center
    boxes_sorted = sorted(zip(merged_boxes, centers), key=lambda bc: bc[1][1])
    ys = [c[1] for _, c in boxes_sorted]
    min_y, max_y = min(ys), max(ys)
    bins = np.linspace(min_y, max_y, grid_rows + 1)

    rows = {i: [] for i in range(grid_rows)}
    for box, (cx, cy) in boxes_sorted:
        row_idx = np.digitize(cy, bins) - 1
        row_idx = max(0, min(row_idx, grid_rows - 1))
        rows[row_idx].append((box, cx))

    final_boxes = []
    for i in range(grid_rows):
        row_boxes = sorted(rows[i], key=lambda bc: bc[1])
        final_boxes.extend([bc[0] for bc in row_boxes])

    return final_boxes

def extract_sprites(image_path, output_dir, expected=16, buffer=3, min_area=500):
    name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("RGBA")
    alpha = np.array(image.split()[-1])

    # Create a mask from the alpha channel.
    mask = alpha > 16
    # Possibly enlarge structure or iterations if needed
    mask = binary_closing(mask, structure=np.ones((3, 3)), iterations=2)

    labeled = label(mask)
    regions = [r for r in regionprops(labeled) if r.area > min_area]
    regions = sorted(regions, key=lambda r: (r.centroid[0], r.centroid[1]))

    if len(regions) < expected:
        print(f"[{name}] ⚠️ Found only {len(regions)} regions, expected {expected}. Continuing anyway.")

    # Build bounding boxes from each region
    initial_boxes = []
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        box = (
            max(0, minc - buffer),
            max(0, minr - buffer),
            min(image.width, maxc + buffer),
            min(image.height, maxr + buffer)
        )
        initial_boxes.append(box)

    # Merge them iteratively
    merged_boxes = merge_all_close_boxes(initial_boxes, dist=10)

    # Remove any smaller boxes fully contained by a bigger box
    filtered_boxes = filter_contained_boxes(merged_boxes)

    # Reorder boxes into a grid
    final_boxes = reorder_boxes_grid(filtered_boxes, expected)

    # Draw and save
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    saved = 0
    for box in final_boxes:
        if saved >= expected:
            break
        x0, y0, x1, y1 = box
        cropped = image.crop(box)
        cleaned = remove_white_border(cropped)
        cleaned.save(os.path.join(output_dir, f"{name}_{saved}.png"))
        draw.rectangle(box, outline="purple", width=2)
        draw.text((x0 + 2, y0 + 2), f"{saved}", fill="purple")
        saved += 1

    debug_path = os.path.join(output_dir, f"{name}_debug.png")
    debug_img.save(debug_path)
    print(f"✅ {name}: Saved {saved} sprites, debug overlay → {debug_path}")

def main():
    folder = os.getcwd()
    output_dir = os.path.join(folder, "output_sprites")
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".webp")):
            extract_sprites(os.path.join(folder, file), output_dir)

if __name__ == "__main__":
    main()
