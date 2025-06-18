import os
import csv
import cv2
import math
import json
import numpy as np


def write_ls_export_json_into_structured(in_json_file, out_json_file):
    with open(in_json_file, 'r') as f:
        contents = json.load(f)
        fw = open(out_json_file, 'w')
        json.dump(contents, fw, indent=4)
        fw.close()


def get_all_key_frame_id_from_video_anno(anno_result):
    # anno_result = contents[0]["annotations"][0]["result"]
    frame_ids = []
    for objects in anno_result:  # loop through all labelled objects
        anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
        for entry in anno_value["sequence"]:  # loop through all labelled key frames
            fid = entry['frame']
            if fid not in frame_ids:
                frame_ids.append(fid)
    return frame_ids


def get_boxes_from_anno_json(anno_file, img_h, img_w, task_id=0):
    # img_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'ch3_250109090000_005200_0001.png')
    # anno_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'sample_annotations.json')
    # image = cv2.imread(img_file)
    # img_h, img_w = image.shape[:2]
    input_boxes = []
    labels = []
    with open(anno_file, 'r') as f:
        annotations = json.load(f)  # tasks
        task = annotations[task_id]  # data in the first labelling task
        objects = task['annotations'][0]['result']  # annotated boxes
        for obj in objects:
            values = obj['value']['sequence'][0]
            x0 = int(values['x'] * img_w / 100.)
            y0 = int(values['y'] * img_h / 100.)
            x1 = int(x0 + values['width'] * img_w / 100.)
            y1 = int(y0 + values['height'] * img_h / 100.)
            # input box in xyxy format
            input_boxes.append([x0, y0, x1, y1])
            labels.append(obj['value']['labels'][0])  # str
    input_boxes = np.array(input_boxes)
    print(input_boxes.shape)
    return input_boxes, labels


def get_frame_cv2boxes_from_video_anno(anno_result, frame, img_h, img_w):
    # anno_result = contents[0]["annotations"][0]["result"]
    # return axis-aligned box in x0y0x1y1 format
    frame_boxes = []
    labels = []
    for objects in anno_result:  # loop through all labelled objects
        anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
        label = anno_value["labels"][0]  # str
        for entry in anno_value["sequence"]:  # loop through all labelled key frames
            frame_id = entry['frame']
            if frame_id == frame:  # be careful about frame ID indexing
                x0 = int(entry['x'] * img_w / 100.)
                y0 = int(entry['y'] * img_h / 100.)
                x1 = int(x0 + entry['width'] * img_w / 100.)
                y1 = int(y0 + entry['height'] * img_h / 100.)
                frame_boxes.append([x0, y0, x1, y1])
                labels.append(label)
                break
    return frame_boxes, labels


def get_frame_cv2obbs_from_video_anno(anno_result, frame, img_h, img_w):
    # label-studio OBB uses: top left corner and width & height in percentage of image size, angle clockwise [0, 360]
    # opencv OBB: centre point, width, height in pixels; angle [-180, 180]
    # get all OBBs in opencv format
    # anno_result = contents[0]["annotations"][0]["result"]
    frame_boxes = []
    labels = []
    for objects in anno_result:  # loop through all labelled objects
        anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
        label = anno_value["labels"][0]  # str
        for entry in anno_value["sequence"]:  # loop through all labelled key frames
            frame_id = entry['frame']
            if frame_id == frame:  # be careful about frame ID indexing
                angle = int(entry['rotation'] - 180)
                cx = entry['x'] * img_w / 100.
                cy = entry['y'] * img_h / 100.
                w = int(entry['width'] * img_w / 100.)
                h = int(entry['height'] * img_h / 100.)
                cx = int(cx + w/2.)
                cy = int(cy + h/2.)
                frame_boxes.append([cx, cy, w, h, angle])
                labels.append(label)
                break
    return frame_boxes, labels


def get_frame_keypoints_from_task_anno(anno_result):
    # keypoints: snout, tailbase, spine_centre, left_ear, right_ear
    keypoints, kp_labels = [], []
    # kp_count = 0
    for kp in anno_result[0]["result"]:
        img_w, img_h = kp["original_width"], kp["original_height"]
        value = kp["value"]  # dict
        x = value["x"] * img_w / 100. # float
        y = value["y"] * img_h / 100.  # float
        label = value["keypointlabels"][0]  # list of str
        # print(f'x: {x:.3f}, y: {y:.3f}, label: {label}')
        keypoints.append([x, y])
        kp_labels.append(label)
        # kp_count += 1
    return keypoints, kp_labels


def get_cv2_box_from_ls_obb(x, y, width, height, angle, img_w, img_h):
    # label-studio OBB uses: centre point and width & height in percentage of image size, angle [0, 360]
    # opencv OBB: centre point, width, height in pixels; angle [-180, 180]
    w = width / 100 * img_w
    h = height / 100 * img_h
    cx = x / 100 * img_w + (w / 2)
    cy = y / 100 * img_h + (h / 2)
    angle = angle - 180 if angle > 180 else angle
    obox = cv2.boxPoints(((cx, cy), (w, h), angle))
    obox = obox.astype(np.int32)
    return obox



def get_ls_obb_from_mask(mask, img_w, img_h):
    # https://labelstud.io/guide/export
    # https://labelstud.io/templates/video_object_detector
    # label-studio OBB: top left corner in percentage of image size; angle [0, 360]
    # label-studio OBB: centre point and width & height in percentage of image size, angle [0, 360]
    mask = mask.astype(np.uint8) * 255  # mask should be binary
    mask = mask.reshape(img_h, img_w, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Combine all points from all contours
    all_points = np.vstack([cnt for cnt in contours])
    rect = cv2.minAreaRect(all_points)
    # opencv OBB: centre point, width, height in pixels; angle [-180, 180]
    (cx, cy), (w, h), angle = rect
    # Adjust angle
    if angle < 0:
        angle += 180
    # Convert center to top-left corner
    x = cx - (w / 2)
    y = cy - (h / 2)
    x = (x / img_w) * 100
    y = (y / img_h) * 100
    width = (w / img_w) * 100
    height = (h / img_h) * 100
    return x, y, width, height, angle


def calc_obb_from_mask_cv2(mask, img_w, img_h):
    # input: binary mask of an object instance
    # output: OBB in opencv format
    mask = mask.astype(np.uint8) * 255  # mask should be binary
    mask = mask.reshape(img_h, img_w, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Combine all points from all contours
    all_points = np.vstack([cnt for cnt in contours])
    rect = cv2.minAreaRect(all_points)
    # opencv OBB: centre point, width, height in pixels; angle [-180, 180]
    (cx, cy), (obw, obh), obr = rect
    return cx, cy, obw, obh, obr


def rotate_point(px, py, cx, cy, angle_deg):
    """
    Rotate point (px, py) around center (cx, cy) by -angle_deg.
    Returns rotated point.
    """
    angle_rad = math.radians(-angle_deg)  # Negative to reverse rotation
    dx, dy = px - cx, py - cy
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    return rx, ry


def is_point_in_obb(px, py, cx, cy, obw, obh, obr):
    """
    Check if point (px, py) is inside the oriented bounding box (obb).
    obb: dict with keys - center (tuple), width, height, angle
    """
    rx, ry = rotate_point(px, py, cx, cy, obr)
    # Check if inside axis-aligned box
    return -obw/2 <= rx <= obw/2 and -obh/2 <= ry <= obh/2


from config import KP_LABELS

def assign_points_to_boxes(keypoints, kp_labels, obbs, obb_labels):
    """
    Assign keypoints to oriented bounding boxes.
    keypoints: list of dicts with keys - x, y, label
    boxes: list of dicts with keys - center, width, height, angle
    Returns a dict: {box_index: [keypoints]}
    """
    assignments = {}
    point = [-1, -1, 'invisible']
    for i, (box, oblb) in enumerate(zip(obbs, obb_labels)):
        k = 'p' + str(oblb)
        assignments[k] = {}
        assignments[k]["OBB"] = box
        for kp in KP_LABELS:
            assignments[k][kp] = point  # set all 5 points in each object as invisible

    for kp, kplb in zip(keypoints, kp_labels):
        px, py = kp[0], kp[1]
        kpl, kpv = kplb#.split(' ')
        for pi, pv in assignments.items():
            cx, cy, obw, obh, obr = pv["OBB"]
            if is_point_in_obb(px, py, cx, cy, obw, obh, obr):
                if assignments[pi][kpl][-1] == 'invisible':
                    assignments[pi][kpl][0] = px
                    assignments[pi][kpl][1] = py
                    assignments[pi][kpl][2] = kpv  # visible/occluded
                else:
                    print(f'Duplicated kp: {pi} {kpl} {kpv}')
                # break  # Assign to first box it belongs to (if needed)
    return assignments


def convert_cv2obb_to_ls_obb(cx, cy, obw, obh, obr, img_w, img_h):
    # LS OBB: angle [0,360] clockwise,
    # CV2 OBB: angle [-180,180],
    # Adjust angle
    if obr < 0:
        obr += 180
    # Convert center to top-left corner
    x = cx - (obw / 2)
    y = cy - (obh / 2)
    x = (x / img_w) * 100
    y = (y / img_h) * 100
    width = (obw / img_w) * 100
    height = (obh / img_h) * 100
    return x, y, width, height, obr


def calc_cv2abb_from_lsobb(x0_p, y0_p, w_p, h_p, img_w, img_h):
    # get the corners coordinates
    x1_p = x0_p + w_p
    y1_p = y0_p + h_p
    x0_obb = x0_p / 100 * img_w
    x1_obb = x1_p / 100 * img_w
    y0_obb = y0_p / 100 * img_h
    y1_obb = y1_p / 100 * img_h
    x0_abb = min(x0_obb, x1_obb)
    x1_abb = max(x0_obb, x1_obb)
    y0_abb = min(y0_obb, y1_obb)
    y1_abb = max(y0_obb, y1_obb)
    return (x0_obb, y0_obb, x1_obb, y1_obb), (x0_abb, y0_abb, x1_abb, y1_abb)


def separate_annotation_per_video(bulk_json_file):
    json_dir = os.path.dirname(bulk_json_file)
    with open(bulk_json_file, 'r') as f:
        annotations = json.load(f)  # tasks
        for anno in annotations:
            video_name = os.path.basename(anno['data']['video'])
            video_name = video_name.split('.')[0]
            save_json = os.path.join(json_dir, video_name + '.json')
            with open(save_json, 'w') as fw:
                json.dump(anno, fw, indent=4)
            print(f'JSON for video saved as {save_json}')


def write_ls_video_export_json_into_mot_csv(ls_json, out_file, img_w, img_h, obj=-1, max_frame=-1, spec_frame=-1):
    # MOT format:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    # label-studio OBB: top left corner, width & height in percentage of image size, angle clockwise [0, 360]
    # header = ['frame', 'id', '', '', '', '', '', '', '', '', '', ]
    fw = open(out_file, 'w')
    writer = csv.writer(fw, delimiter=',', quotechar='"')
    with open(ls_json, 'r') as f:
        contents = json.load(f)
        anno_result = contents["annotations"][0]["result"]
        for objects in anno_result:  # loop through all labelled objects
            anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
            label = anno_value["labels"][0]  # str
            obj_id = int(label.replace('p', ''))
            if 0 < obj != obj_id:
                continue
            for entry in anno_value["sequence"]:  # loop through all labelled key frames
                frame_id = int(entry['frame'])
                if 0 < max_frame < frame_id:
                    break
                if 0 < spec_frame != frame_id:
                    continue
                # rotation parameter required here, because xywh are rotated box measures
                angle_deg = int(entry['rotation'])
                xc = entry['x'] * img_w / 100.
                yc = entry['y'] * img_h / 100.
                w = entry['width'] * img_w / 100.
                h = entry['height'] * img_h / 100.
                dx, dy = w / 2, h / 2
                xc += dx
                yc += dy
                angle_rad = math.radians(angle_deg)
                corners = []
                for sign_x, sign_y in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                    x = sign_x * dx
                    y = sign_y * dy
                    # Rotate
                    x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
                    y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
                    corners.append([int(xc + x_rot), int(yc + y_rot)])
                # x0_abb = min(x0, x1)
                # y0_abb = min(y0, y1)
                # w_abb = max(x0, x1) - x0_abb
                # h_abb = max(y0, y1) - y0_abb
                # writer.writerow([frame_id, obj_id, x0_abb, y0_abb, w_abb, h_abb, -1, -1, -1, -1])
                writer.writerow([frame_id, obj_id, corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                                 corners[2][0], corners[2][1], corners[3][0], corners[3][1]])
                if 0 < spec_frame == frame_id:
                    break
    fw.close()