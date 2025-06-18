import cv2
import numpy as np


def get_min_area_rect_from_mask_array(mask, img_h, img_w):
    mask = mask.astype(np.uint8) * 255  # mask should be binary
    mask = mask.reshape(img_h, img_w, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Combine all points from all contours
    all_points = np.vstack([cnt for cnt in contours])
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    return box


def binary_mask_to_rle(mask):
    """
    Convert binary mask (2D NumPy array) to RLE.
    Returns RLE as list of start positions and lengths.
    """
    pixels = mask.flatten(order='F')  # Flatten in column-major order (COCO standard)
    pixels = np.concatenate([[0], pixels, [0]])  # Add sentinel values
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Find where values change
    runs[1::2] -= runs[::2]  # Compute lengths
    return runs.tolist()


def draw_keypoints_obb_assignments_on_image(image, assignments, box_colors):
    for pi, pv in assignments.items():
        cx, cy, obw, obh, obr = pv["OBB"]
        rect = ((float(cx), float(cy)), (obw, obh), float(obr))
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)
        bcolor = box_colors[pi]
        cv2.drawContours(image, [box_points], 0, color=bcolor, thickness=3)
        for kpl, kpv in pv.items():
            if kpl == "OBB":
                continue
            if kpv[0] >= 0 and kpv[1] >= 0:
                cv2.circle(image, (int(kpv[0]), int(kpv[1])), 5, color=bcolor, thickness=-1)
        cv2.putText(image, pi, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)
        cv2.putText(image, pi, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, bcolor, thickness=2)
    return image