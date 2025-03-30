import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing, grey_erosion

def remove_white_border(image, white_threshold=220):
    data = np.array(image)
    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    near_white = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)
    faint = a < 255
    data[near_white & faint] = [0, 0, 0, 0]
    return Image.fromarray(data)

def shave_outline(image, erosion_size=1):
    """
    Erodes the alpha channel by 'erosion_size' pixels using grey_erosion.
    """
    arr = np.array(image)
    alpha = arr[..., 3]
    kernel_size = (2 * erosion_size + 1, 2 * erosion_size + 1)
    alpha_ero = grey_erosion(alpha, size=kernel_size)
    arr[..., 3] = alpha_ero
    return Image.fromarray(arr, mode='RGBA')

def smooth_alpha(image, radius=1):
    """
    Applies a Gaussian blur to the alpha channel then re-runs the white-border removal.
    This helps smooth jagged edges without reintroducing the white outline.
    """
    r, g, b, a = image.split()
    a_blurred = a.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred = Image.merge("RGBA", (r, g, b, a_blurred))
    # Reapply white-border removal to clear out any new white fringe.
    return remove_white_border(blurred)

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
    x0s, y0s, x1s, y1s = box_small
    x0b, y0b, x1b, y1b = box_big
    return (x0b <= x0s) and (y0b <= y0s) and (x1b >= x1s) and (y1b >= y1s)

def filter_contained_boxes(boxes):
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
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    if xA2 < xB1:
        gap_x = xB1 - xA2
    elif xB2 < xA1:
        gap_x = xA1 - xB2
    else:
        gap_x = 0

    if yA2 < yB1:
        gap_y = yB1 - yA2
    elif yB2 < yA1:
        gap_y = yA1 - yB2
    else:
        gap_y = 0

    return (gap_x <= dist) and (gap_y <= dist)

def merge_all_close_boxes(boxes, dist=10):
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
    if not merged_boxes:
        return merged_boxes

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in merged_boxes]
    grid_rows = int(round(expected ** 0.5)) or 1

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

def extract_sprites(image_path, output_dir, expected=16, buffer=3, min_area=500,
                    remove_outline=False, erosion_size=1, smooth=False, smooth_radius=1):
    name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("RGBA")
    alpha = np.array(image.split()[-1])

    # Create a mask from the alpha channel.
    mask = alpha > 16
    mask = binary_closing(mask, structure=np.ones((3, 3)), iterations=2)

    labeled = label(mask)
    regions = [r for r in regionprops(labeled) if r.area > min_area]
    regions = sorted(regions, key=lambda r: (r.centroid[0], r.centroid[1]))

    if len(regions) < expected:
        print(f"[{name}] ⚠️ Found only {len(regions)} regions, expected {expected}. Continuing anyway.")

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

    merged_boxes = merge_all_close_boxes(initial_boxes, dist=10)
    filtered_boxes = filter_contained_boxes(merged_boxes)
    final_boxes = reorder_boxes_grid(filtered_boxes, expected)

    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    saved = 0

    for box in final_boxes:
        if saved >= expected:
            break
        x0, y0, x1, y1 = box
        cropped = image.crop(box)
        cleaned = remove_white_border(cropped)

        if remove_outline:
            cleaned = shave_outline(cleaned, erosion_size=erosion_size)
            if smooth:
                cleaned = smooth_alpha(cleaned, radius=smooth_radius)

        out_name = f"{name}_{saved}.png"
        cleaned.save(os.path.join(output_dir, out_name))

        draw.rectangle(box, outline="purple", width=2)
        draw.text((x0 + 2, y0 + 2), f"{saved}", fill="purple")
        saved += 1

    debug_path = os.path.join(output_dir, f"{name}_debug.png")
    debug_img.save(debug_path)
    print(f"✅ {name}: Saved {saved} sprites, debug overlay → {debug_path}")

def main(args):
    folder = os.getcwd()
    output_dir = os.path.join(folder, "output_sprites")
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".webp")):
            extract_sprites(
                image_path=os.path.join(folder, file),
                output_dir=output_dir,
                expected=args.expected,
                buffer=args.buffer,
                min_area=args.min_area,
                remove_outline=args.remove_outline,
                erosion_size=args.erosion_size,
                smooth=args.smooth,
                smooth_radius=args.smooth_radius
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sprites with optional outline removal and smoothing.")
    parser.add_argument("--remove-outline", action="store_true",
                        help="Erode alpha to remove the outer halo.")
    parser.add_argument("--expected", type=int, default=16,
                        help="Expected number of sprites.")
    parser.add_argument("--buffer", type=int, default=3,
                        help="Buffer (in pixels) around each bounding box.")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Minimum area for a region to be considered.")
    parser.add_argument("--erosion-size", type=int, default=1,
                        help="Erosion size for alpha channel erosion (default: 1).")
    parser.add_argument("--smooth", action="store_true",
                        help="Apply Gaussian smoothing to the alpha channel after erosion.")
    parser.add_argument("--smooth-radius", type=float, default=1,
                        help="Gaussian blur radius for smoothing (default: 1).")
    args = parser.parse_args()
    main(args)
