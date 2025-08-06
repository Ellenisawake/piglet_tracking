import os
import csv
import cv2
import math
import json
import glob
import numpy as np
from datetime import datetime


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


def old_write_ls_video_export_json_into_mot_csv(ls_json, out_file, img_w, img_h, obj=-1, max_frame=-1, spec_frame=-1):
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
                corners = []
                # rotation parameter required here, because xywh are rotated box measures
                angle_deg = int(entry['rotation'])
                xtl = entry['x'] * img_w / 100.  # top left
                ytl = entry['y'] * img_h / 100.
                corners.append(xtl)
                corners.append(ytl)
                w = entry['width'] * img_w / 100.
                h = entry['height'] * img_h / 100.
                # dx, dy = w / 2, h / 2
                # how to obtain the correct box centre ???
                # xc = xtl + dx
                # yc = ytl + dy
                angle_rad = math.radians(angle_deg)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                # top right
                P1 = (xtl + w * cos_t, ytl + w * sin_t)
                corners.append(P1[0])
                corners.append(P1[1])
                # bottom right
                P2 = (xtl + w * cos_t - h * sin_t, ytl + w * sin_t + h * cos_t)
                corners.append(P2[0])
                corners.append(P2[1])
                # bottom left
                P3 = (xtl - h * sin_t, ytl + h * cos_t)
                corners.append(P3[0])
                corners.append(P3[1])
                # for sign_x, sign_y in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                #     x = sign_x * dx
                #     y = sign_y * dy
                #     # Rotate
                #     x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
                #     y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
                #     corners.append([int(xc + x_rot), int(yc + y_rot)])
                # x0_abb = min(x0, x1)
                # y0_abb = min(y0, y1)
                # w_abb = max(x0, x1) - x0_abb
                # h_abb = max(y0, y1) - y0_abb
                # writer.writerow([frame_id, obj_id, x0_abb, y0_abb, w_abb, h_abb, -1, -1, -1, -1])
                corners = [int(c) for c in corners]
                writer.writerow([frame_id, obj_id, corners[0], corners[1], corners[2], corners[3], corners[4],
                                 corners[5], corners[6], corners[7]])
                if 0 < spec_frame == frame_id:
                    break
    fw.close()



def write_ls_video_export_json_into_mot_csv_fps25_to30(ls_json, out_file, img_w, img_h, obj=-1, max_frame=-1, spec_frame=-1):
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
                xs, ys = [], []
                # rotation parameter required here, because xywh are rotated box measures
                angle_deg = int(entry['rotation'])
                xtl = entry['x'] * img_w / 100.  # top left
                ytl = entry['y'] * img_h / 100.
                xs.append(xtl)
                ys.append(ytl)
                w = entry['width'] * img_w / 100.
                h = entry['height'] * img_h / 100.
                angle_rad = math.radians(angle_deg)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                # top right
                P1 = (xtl + w * cos_t, ytl + w * sin_t)
                xs.append(P1[0])
                ys.append(P1[1])
                # bottom right
                P2 = (xtl + w * cos_t - h * sin_t, ytl + w * sin_t + h * cos_t)
                xs.append(P2[0])
                ys.append(P2[1])
                # bottom left
                P3 = (xtl - h * sin_t, ytl + h * cos_t)
                xs.append(P3[0])
                ys.append(P3[1])
                xs = [int(c) for c in xs]
                ys = [int(c) for c in ys]
                # top left corner
                x0, y0 = min(xs), min(ys)
                # bottom right corner
                x1, y1 = max(xs), max(ys)
                w_abb, h_abb = x1 - x0, y1 - y0
                frame_to_write = int(float(frame_id) / 25. * 30.)
                # # write the 4 corners for debugging
                # writer.writerow([frame_to_write, obj_id, xs[0], ys[0], xs[1], ys[1], xs[2],
                #                  ys[2], xs[3], ys[3]])
                # MOT format:
                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                writer.writerow([frame_to_write, obj_id, x0, y0, w_abb, h_abb, -1, -1, -1, -1])
                if 0 < spec_frame == frame_id:
                    break
    fw.close()


def write_ls_video_obb_export_json_into_mot_abb_csv(ls_json, out_file, img_w, img_h):#, obj=-1, max_frame=-1, spec_frame=-1):
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
            # if 0 < obj != obj_id:
            #     continue
            for entry in anno_value["sequence"]:  # loop through all labelled key frames
                frame_id = int(entry['frame'])
                # if 0 < max_frame < frame_id:
                #     break
                # if 0 < spec_frame != frame_id:
                #     continue
                xs, ys = [], []
                # rotation parameter required here, because xywh are rotated box measures
                angle_deg = int(entry['rotation'])
                xtl = entry['x'] * img_w / 100.  # top left
                ytl = entry['y'] * img_h / 100.
                xs.append(xtl)
                ys.append(ytl)
                w = entry['width'] * img_w / 100.
                h = entry['height'] * img_h / 100.
                angle_rad = math.radians(angle_deg)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                # top right
                P1 = (xtl + w * cos_t, ytl + w * sin_t)
                xs.append(P1[0])
                ys.append(P1[1])
                # bottom right
                P2 = (xtl + w * cos_t - h * sin_t, ytl + w * sin_t + h * cos_t)
                xs.append(P2[0])
                ys.append(P2[1])
                # bottom left
                P3 = (xtl - h * sin_t, ytl + h * cos_t)
                xs.append(P3[0])
                ys.append(P3[1])
                xs = [int(c) for c in xs]
                ys = [int(c) for c in ys]
                # top left corner
                x0, y0 = min(xs), min(ys)
                # bottom right corner
                x1, y1 = max(xs), max(ys)
                w_abb, h_abb = x1 - x0, y1 - y0
                # MOT format:
                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                writer.writerow([frame_id, obj_id, x0, y0, w_abb, h_abb, -1, -1, -1, -1])
                # if 0 < spec_frame == frame_id:
                #     break
    fw.close()


def write_ls_video_export_json_into_yolo_obb(ls_json, out_dir, image_dir, img_w, img_h, spec_frame=-1):#):#, obj=-1, max_frame=-1
    '''
    YOLO OBB GT format designates bounding boxes by four corner points with coordinates normalized between 0 and 1
    class_index x1 y1 x2 y2 x3 y3 x4 y4
    '''
    # label-studio OBB: top left corner, width & height in percentage of image size, angle clockwise [0, 360]
    # header = ['frame', 'id', '', '', '', '', '', '', '', '', '', ]

    json_reader = open(ls_json, 'r')
    contents = json.load(json_reader)
    json_reader.close()
    anno_result = contents["annotations"][0]["result"]
    video_file = contents["data"]["video"]
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    # file = os.path.join(image_dir, f'{video_name}_{spec_frame:04d}.jpg')
    label_file_count = 0
    for file in sorted(glob.glob(os.path.join(image_dir, f'{video_name}_*.jpg'))):
        file_name = os.path.basename(file)
        file_name = os.path.splitext(file_name)[0]
        frame_id = int(file_name.split('_')[-1])
        if 0 < spec_frame != frame_id:
            continue
        # check image file existing and copy file name for GT file name
        sample_name = f'{video_name}_{frame_id:04d}'
        out_file = os.path.join(out_dir, f'{sample_name}.txt')
        gt_objects = []
        # loop through all labelled objects because LS JSON is organised by objects
        for objects in anno_result:
            anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
            label = anno_value["labels"][0]  # str, not used here in detection GT
            for entry in anno_value["sequence"]:  # loop through all labelled key frames
                frame_json = int(entry['frame'])
                # find object json entry for this frame
                if frame_json != frame_id:
                    continue
                # rotation parameter required here, because xywh are rotated box measures
                angle_deg = int(entry['rotation'])
                xtl = entry['x'] * img_w / 100.  # top left
                ytl = entry['y'] * img_h / 100.
                w = entry['width'] * img_w / 100.
                h = entry['height'] * img_h / 100.
                angle_rad = math.radians(angle_deg)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                # top right
                P1 = (xtl + w * cos_t, ytl + w * sin_t)
                # bottom right
                P2 = (xtl + w * cos_t - h * sin_t, ytl + w * sin_t + h * cos_t)
                # bottom left
                P3 = (xtl - h * sin_t, ytl + h * cos_t)
                # class_index x1 y1 x2 y2 x3 y3 x4 y4 (normalized between 0 and 1)
                x1, x2, x3, x4 = xtl / img_w, P1[0] / img_w, P2[0] / img_w, P3[0] / img_w
                y1, y2, y3, y4 = ytl / img_h, P1[1] / img_h, P2[1] / img_h, P3[1] / img_h
                gt_objects.append([x1, y1, x2, y2, x3, y3, x4, y4])
                # end loop within object when this frame found
                if frame_json == frame_id:
                    break
        with open(out_file, 'w') as writer:
            for entry in gt_objects:
                writer.write('0')
                for v in entry:
                    writer.write(f'\t{v:.6f}')
                writer.write('\n')
                writer.flush()
        print(f'Finished writing {out_file}')
        label_file_count += 1
        # end loop through all image files when specific frame found
        if spec_frame > 0 and spec_frame == frame_id:
            break
    return label_file_count


def write_ls_image_export_json_into_yolo_obb(ls_json, out_dir, img_w, img_h):
    json_reader = open(ls_json, 'r')
    contents = json.load(json_reader)  # list of tasks per image
    json_reader.close()
    json_name = os.path.splitext(os.path.basename(ls_json))[0]
    labels_list_file = os.path.join(os.path.dirname(ls_json), f'{json_name}_labels_list.txt')
    labels_list = []
    for i, tasks in enumerate(contents):
        image_name = os.path.basename(tasks["data"]["image"])

        # check that yolo prediction has been manually corrected for the image
        # Jian: is this correct?
        created_timestamp, updated_timestamp = tasks["created_at"], tasks["updated_at"]
        created_timestamp = created_timestamp.split('.')[0]  # 2025-07-31T10:08:05
        updated_timestamp = updated_timestamp.split('.')[0]
        if created_timestamp == updated_timestamp:
            print(f'Skipping {image_name} because predictions not verified')
            continue

        # check image file existing and copy file name for GT file name
        sample_name = os.path.splitext(image_name)[0]
        out_file = os.path.join(out_dir, f'{sample_name}.txt')
        gt_objects = []
        exclude_image = False
        for object in tasks["annotations"]["result"]:
            # choice tag, not using at the moment
            # if object["type"] == 'choices':
            #     choice = object["value"]["choices"][0]  # Include or Exclude
            #     if choice == 'Exclude':
            #         exclude_image = True
            #         break
            if object["type"] == 'rectanglelabels':
                xtl = object["value"]["x"] * img_w / 100.  # top left
                ytl = object["value"]["y"] * img_h / 100.
                w = object["value"]["width"] * img_w / 100.
                h = object["value"]["height"] * img_h / 100.
                r = object["value"]["rotation"]
                angle_rad = math.radians(r)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                # top right
                P1 = (xtl + w * cos_t, ytl + w * sin_t)
                # bottom right
                P2 = (xtl + w * cos_t - h * sin_t, ytl + w * sin_t + h * cos_t)
                # bottom left
                P3 = (xtl - h * sin_t, ytl + h * cos_t)
                # class_index x1 y1 x2 y2 x3 y3 x4 y4 (normalized between 0 and 1)
                x1, x2, x3, x4 = xtl / img_w, P1[0] / img_w, P2[0] / img_w, P3[0] / img_w
                y1, y2, y3, y4 = ytl / img_h, P1[1] / img_h, P2[1] / img_h, P3[1] / img_h
                gt_objects.append([x1, y1, x2, y2, x3, y3, x4, y4])

        if exclude_image:
            print(f'Skipping {image_name} because tagged as exclude')
            continue

        with open(out_file, 'w') as writer:
            for entry in gt_objects:
                writer.write('0')
                for v in entry:
                    writer.write(f'\t{v:.6f}')
                writer.write('\n')
                writer.flush()
        print(f'Finished writing {out_file}')
        labels_list.append(sample_name)
    np.savetxt(labels_list_file, labels_list, fmt='%s')
    return len(labels_list)



def linear_interpolate(a, b, t):
    return a + (b - a) * t


def linear_interpolate_angle(a, b, t):
    angle_diff = b - a
    angle_diff = ((angle_diff + 180) % 360) - 180
    angle = a + angle_diff * t
    return ((angle + 180) % 360) - 180


def yolo_to_labelstudio_rotation_angle(yolo_angle_rad):
    # # # Shift base from +x (YOLO) to -y (Label Studio)
    # shifted = yolo_angle_rad - math.pi / 2  # Shift so 0 rad points "up"
    # angle_deg = (360 - math.degrees(shifted)) % 360
    angle_deg = math.degrees(yolo_angle_rad)
    # if angle_deg < 0:
    # angle_deg = (360 - angle_deg) % 360
    return angle_deg


def yolo_obb_to_labelstudio_obb_incomplete(yolo_obb, image_width, image_height):
    cx, cy, width, height, angle_rad = yolo_obb
    angle = float(math.degrees(angle_rad))

    # Offset from center to top-left (before rotation)
    dx = -width / 2
    dy = -height / 2
    # Rotate offset clockwise by theta
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)
    dx_rot = dx * cos_t + dy * sin_t
    dy_rot = -dx * sin_t + dy * cos_t
    # Calculate top-left corner after rotation
    xtl = cx + dx  # + dx_rot
    ytl = cy + dy  # + dy_rot

    x = float(xtl / image_width * 100.0)
    y = float(ytl / image_height * 100.0)
    w = float(width / image_width * 100.0)
    h = float(height / image_height * 100.0)
    return x, y, w, h, angle


# https://github.com/cfrancois7/label-studio-sdk/blob/feat/import_yolo_obb/src/label_studio_sdk/converter/utils.py#L481
def yolo_obb_to_labelstudio_obb(yolo_obb, image_width, image_height):
    # expects xyxyxyxy style yolo result (4*2) in pixels
    # x1, y1, x2, y2, x3, y3, x4, y4 = yolo_obb
    center_x = np.mean(yolo_obb[:, 0])
    center_y = np.mean(yolo_obb[:, 1])
    width = np.linalg.norm(yolo_obb[0] - yolo_obb[1])
    height = np.linalg.norm(yolo_obb[0] - yolo_obb[3])
    dx = yolo_obb[1, 0] - yolo_obb[0, 0]
    dy = yolo_obb[1, 1] - yolo_obb[0, 1]
    r = np.degrees(np.arctan2(dy, dx))
    top_left_x = (
            center_x
            - (width / 2) * np.cos(np.radians(r))
            + (height / 2) * np.sin(np.radians(r))
    )
    top_left_y = (
            center_y
            - (width / 2) * np.sin(np.radians(r))
            - (height / 2) * np.cos(np.radians(r))
    )

    # dx_w = x2 - x1
    # dy_w = y2 - y1
    # width = math.hypot(dx_w, dy_w)
    #
    # # Compute height: distance from point 1 to 4
    # dx_h = x4 - x1
    # dy_h = y4 - y1
    # height = math.hypot(dx_h, dy_h)
    #
    # # Compute rotation: angle from +x axis to vector (x2 - x1, y2 - y1)
    # angle_rad = math.atan2(dy_w, dx_w)
    # angle_deg = math.degrees(angle_rad) % 360  # CW from +x

    # Normalize the values
    x = float(top_left_x / image_width * 100)
    y = float(top_left_y / image_height) * 100
    w = float(width / image_width * 100)
    h = float(height / image_height * 100)
    return x, y, w, h, float(r)


def interpolate_obbs(sequence, total_frames, min_life=1):
    """
    sequence: list of keyframes with OBB info
    total_frames: int, total frame count of the video
    min_life: int, minimum number of appearing key frames to count object as valid
    Returns: dict of {frame_number: interpolated OBB dict}
    """
    interpolated_sequence = []

    # Sort by frame index
    sequence = sorted(sequence, key=lambda k: k["frame"])
    num_keyframes = len(sequence)
    # Jian: filter out object which only appears on one frame - mistakenly clicked
    if num_keyframes < min_life:
        return interpolated_sequence

    for idx in range(num_keyframes - 1):
        start = sequence[idx]
        end = sequence[idx + 1]
        # Jian: need to check flag here to identify broken tracklet --
        # the consequent frames interpolation is toggled off, object occluded/OoV
        # we need to decide whether to propagate box from previous frame or break tracklet
        if start["enabled"] is False:  # == "true":
            continue
        start_frame = int(start["frame"])
        end_frame = int(end["frame"])
        frame_gap = end_frame - start_frame

        for f in range(start_frame, end_frame):
            t = (f - start_frame) / frame_gap if frame_gap != 0 else 0

            obb = {
                "frame": f,
                "enabled": "true",
                "x": linear_interpolate(start["x"], end["x"], t),
                "y": linear_interpolate(start["y"], end["y"], t),
                "width": linear_interpolate(start["width"], end["width"], t),
                "height": linear_interpolate(start["height"], end["height"], t),
                # rotation interpolation needs to be handled carefully in terms of rotation direction
                "rotation": linear_interpolate_angle(start["rotation"], end["rotation"], t),
                "time": linear_interpolate(start.get("time", 0), end.get("time", 0), t),
            }
            interpolated_sequence.append(obb)

    # Add the last keyframe explicitly
    end = sequence[-1]
    interpolated_sequence.append({
        "frame": end["frame"],
        "enabled": end["enabled"],#
        "x": end["x"],
        "y": end["y"],
        "width": end["width"],
        "height": end["height"],
        "rotation": end["rotation"],
        "time": end.get("time", 0),
    })
    end_frame = int(end["frame"])
    if num_keyframes > 1 and total_frames > end_frame:
        end_time = float(end["time"])
        # when last key frame is not the last frame in the video
        first_key_frame_id = int(sequence[0]["frame"])
        first_key_frame_time = float(sequence[0]["time"])
        # print(f'first_key_frame_id: {first_key_frame_id}, end_key_frame_id: {end_frame}')
        dftime = (end_time - first_key_frame_time) / (end_frame - first_key_frame_id)
        print(f'Key frames ({end_frame}) do not cover all {total_frames} frames')
        if end["enabled"] is True: # == "true":
            print(f'Extending annotations from {end_frame} to {total_frames} frames')
            # frame_gap = total_frames - end_frame
            for f in range(end_frame+1, total_frames+1):
                r = f - end_frame
                interpolated_time = end_time + dftime * r
                interpolated_sequence.append({
                    "frame": f,
                    "enabled": end["enabled"],  # True,
                    "x": end["x"],
                    "y": end["y"],
                    "width": end["width"],
                    "height": end["height"],
                    "rotation": end["rotation"],
                    "time": interpolated_time,
                })
    return interpolated_sequence


def process_labelstudio_interpolation(input_json_path, output_json_path, min_life=1):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    for annotation in annotations:
        # loop through all annotated objects
        for count, obj in enumerate(annotation["result"]):
            value = obj["value"]
            obj_label = value["labels"][0]
            sequence = value["sequence"]
            # # for missing frame 1800 in ch2_20250109090000_000200 only -----
            # annotation["result"][count]["value"]["framesCount"] = 1800
            # # for missing frame 1800 in ch2_20250109090000_000200 only -----
            total_frames = value["framesCount"]
            print(f'{obj_label} total frames: {total_frames} ---')

            interpolated_sequence = interpolate_obbs(sequence, total_frames, min_life)
            # Replace with dense interpolated sequence
            if len(interpolated_sequence) > min_life:
                value["sequence"] = interpolated_sequence

    # check video path, convert to Jian's local -------------------------------
    video_name = os.path.basename(data["data"]["video"])
    video_path = os.path.join("/data/local-files/?d=Documents/Work/QUBRF/Piglets/Data/MSUclips", video_name)
    data["data"]["video"] = video_path
    # check video path, convert to Jian's local -------------------------------

    # Write re-importable JSON
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Interpolated JSON saved to: {output_json_path}")


def write_ls_json_obb_to_obb_tracking_gt_csv(ls_json, out_file, img_w, img_h):
    fw = open(out_file, 'w')
    writer = csv.writer(fw, delimiter=',', quotechar='"')
    with open(ls_json, 'r') as f:
        contents = json.load(f)
        anno_result = contents["annotations"][0]["result"]
        for objects in anno_result:  # loop through all labelled objects
            anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
            label = anno_value["labels"][0]  # str
            obj_id = int(label.replace('p', ''))
            for entry in anno_value["sequence"]:  # loop through all labelled key frames
                frame_id = int(entry['frame'])
                # rotation parameter required here, because xywh are rotated box measures
                angle_deg = int(entry['rotation'])
                # top left
                xtl = entry['x'] * img_w / 100.
                ytl = entry['y'] * img_h / 100.
                w = entry['width'] * img_w / 100.
                h = entry['height'] * img_h / 100.
                angle_rad = math.radians(angle_deg)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                # top right
                x2, y2 = xtl + w * cos_t, ytl + w * sin_t
                # bottom right
                x3, y3 = xtl + w * cos_t - h * sin_t, ytl + w * sin_t + h * cos_t
                # bottom left
                x4, y4 = xtl - h * sin_t, ytl + h * cos_t
                # frame, obj_id, x1 y1 x2 y2 x3 y3 x4 y4 (in pixels)
                x1, y1, x2, y2, x3, y3, x4, y4 = int(xtl), int(ytl), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
                writer.writerow([frame_id, obj_id, x1, y1, x2, y2, x3, y3, x4, y4])
    fw.close()


def write_image_task_json_from_list(image_list, image_folder, json_file_to_write):
    tasks = []
    prefix = f"/data/local-files/?d={image_folder}/"
    for i, image in enumerate(image_list):
        image_task = {}
        image_task["id"] = i + 1
        image_task["data"] = {}
        image_task["data"]["image"] = prefix + image
        tasks.append(image_task)
    with open(json_file_to_write, 'w') as fw:
        json.dump(tasks, fw, indent=4)


def write_image_prediction_json_template_from_image_list(image_list, image_folder, prediction_json_file):
    json_contents = []  # list of all tasks
    prefix = f"/data/local-files/?d={image_folder}/"
    for i, image in enumerate(image_list):
        image_task = {}
        image_task["id"] = i + 1
        image_task["data"] = {}
        image_task["data"]["image"] = prefix + image
        image_task["predictions"] = []  # list of ?
        json_contents.append(image_task)
    with open(prediction_json_file, 'w') as fw:
        json.dump(json_contents, fw, indent=4)
    print(f'Finish saving json at {prediction_json_file}')


def compare_ls_timestamp_and_find_later(frist_timestamp, second_timestamp):
    fmt = "%Y-%m-%dT%H:%M:%S"
    dt1 = datetime.strptime(frist_timestamp, fmt)
    dt2 = datetime.strptime(second_timestamp, fmt)
    return 1 if dt1 > dt2 else 2


def convert_ls_image_export_json_into_yolo_obb_and_copy_images(ls_json, label_save_dir, image_save_dir, image_source_dir, img_w, img_h):
    json_reader = open(ls_json, 'r')
    contents = json.load(json_reader)  # list of tasks per image
    json_reader.close()
    json_name = os.path.splitext(os.path.basename(ls_json))[0]
    labels_list_file = os.path.join(os.path.dirname(ls_json), f'{json_name}_labels_list.txt')
    labels_list = []
    for i, tasks in enumerate(contents):
        image_name = os.path.basename(tasks["data"]["image"])

        # check that yolo prediction has been manually corrected for the image
        # Jian: is this correct?
        created_timestamp, updated_timestamp = tasks["created_at"], tasks["updated_at"]
        created_timestamp = created_timestamp.split('.')[0]  # 2025-07-31T10:08:05
        updated_timestamp = updated_timestamp.split('.')[0]
        if created_timestamp == updated_timestamp:
            print(f'Skipping {image_name} because predictions not verified')
            continue

        # check image file existing and copy file name for GT file name
        sample_name = os.path.splitext(image_name)[0]
        image_save_file = os.path.join(image_save_dir, f'{sample_name}.jpg')
        # image_source_dir = 'WTData/Piglet/Raw/obb_images'
        if not os.path.exists(image_save_file):
            command = f"cp {sample_name}.jpg ../../../Dataset/Piglet2508/batch0805/images/train/"
            os.chdir(image_source_dir)
            os.system(command)
            print(f'Copied {sample_name} into dataset folder')

        out_file = os.path.join(label_save_dir, f'{sample_name}.txt')
        gt_objects = []
        exclude_image = False
        for object in tasks["annotations"][0]["result"]:
            # choice tag, not using at the moment
            # if object["type"] == 'choices':
            #     choice = object["value"]["choices"][0]  # Include or Exclude
            #     if choice == 'Exclude':
            #         exclude_image = True
            #         break
            if object["type"] == 'rectanglelabels':
                xtl = object["value"]["x"] * img_w / 100.  # top left
                ytl = object["value"]["y"] * img_h / 100.
                w = object["value"]["width"] * img_w / 100.
                h = object["value"]["height"] * img_h / 100.
                r = object["value"]["rotation"]
                angle_rad = math.radians(r)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                # top right
                P1 = (xtl + w * cos_t, ytl + w * sin_t)
                # bottom right
                P2 = (xtl + w * cos_t - h * sin_t, ytl + w * sin_t + h * cos_t)
                # bottom left
                P3 = (xtl - h * sin_t, ytl + h * cos_t)
                # class_index x1 y1 x2 y2 x3 y3 x4 y4 (normalized between 0 and 1)
                x1, x2, x3, x4 = xtl / img_w, P1[0] / img_w, P2[0] / img_w, P3[0] / img_w
                y1, y2, y3, y4 = ytl / img_h, P1[1] / img_h, P2[1] / img_h, P3[1] / img_h
                '''
                YOLO OBB GT format designates bounding boxes by four corner points with coordinates normalized between 0 and 1
                class_index x1 y1 x2 y2 x3 y3 x4 y4
                '''
                gt_objects.append([x1, y1, x2, y2, x3, y3, x4, y4])

        if exclude_image:
            print(f'Skipping {image_name} because tagged as exclude')
            continue

        with open(out_file, 'w') as writer:
            for entry in gt_objects:
                writer.write('0')
                for v in entry:
                    writer.write(f'\t{v:.6f}')
                writer.write('\n')
                writer.flush()
        print(f'Finished writing {out_file}')
        labels_list.append(sample_name)
    np.savetxt(labels_list_file, labels_list, fmt='%s')
    return len(labels_list)

