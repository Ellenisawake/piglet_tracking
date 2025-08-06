import os
import cv2
import csv
import glob
import json
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
import motmetrics as mm

from collections import defaultdict

import ls_io_utils
from config import ROOT_DIR, BOX_COLORS
import eval_utils
import io_utils

'''
steps to get key points from obb
    run yolo-obb to get obb
    rotate the image using angle from obb, so that obb becomes axis-aligned
    find coordinates of corners of the obb on the rotated image
    crop axis-aligned box to get object patch
    input patch to yolo-pose to get key points

OBB dataset format can be found in detail in the Dataset Guide. 
The YOLO OBB format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1
class_index x1 y1 x2 y2 x3 y3 x4 y4
'''


def run_yolo_obb():
    test_frames = [48, 600, ]#,
    video_name = 'ch1_20250109090000_000100'
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet')
    visuals_dir = os.path.join(data_dir, 'yolo11x_obb')
    model_file = os.path.join(data_dir, 'yolo11x_obb', 'best_1.pt')
    # Load a model
    model = YOLO(model_file)  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    # # Train the model
    # results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
    for n in test_frames:
        test_image_file = os.path.join(data_dir, video_name, f'frame_{n:04d}.png')
        results = model(test_image_file)  # predict on an image  # gpu tensor
        image_bgr = cv2.imread(test_image_file)
        print(image_bgr.shape)
        # Access the results
        names = dict(results[0].names)
        # result = results[0].obb.cpu().numpy()
        locations, classes = results[0].obb.xywhr.cpu().numpy(), results[0].obb.cls.cpu().numpy()
        for loc, cls in zip(locations, classes):
            name = names[cls]
            x_center, y_center, width, height, angle = loc#.xywhr[0]
            x_center, y_center = int(x_center), int(y_center)
            angle = np.degrees(angle)
            # OpenCV expects angle in degrees, with clockwise positive
            # rect = ((x_center, y_center), (width, height), 0)#angle)
            # four vertices are returned in clockwise order starting from the point with greatest y
            # the rightmost is the starting point if two points with same greatest y
            # box = cv2.boxPoints(rect)
            box = cv2.boxPoints(((x_center, y_center), (width, height), angle))
            if box is None or len(box) == 0:
                print("No cv2 boxes returned")
                continue  # Skip invalid boxes
            if np.any(box < 0):
                print("Skipping box with negative coordinates:", box)
                continue

            box = box.astype(np.int32)
            cv2.drawContours(image_bgr, [box], -1, color=(255,255,255), thickness=2)
            cv2.putText(image_bgr, name, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(128, 128, 128), thickness=3)
            cv2.putText(image_bgr, name, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 255, 255), thickness=2)
            # break
        result_img_file = os.path.join(visuals_dir, f'{video_name}_yolo11x_obb_frame{n}_pen.png')
        image_bgr = image_bgr[:, 800:2050, :]
        cv2.putText(image_bgr, f'frame{n}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                    color=(255, 255, 255), thickness=3)
        cv2.imwrite(result_img_file, image_bgr)
        print(f'Finish writing {result_img_file}')


def obb_to_aabb(obb):
    """
    Convert an oriented bounding box to an axis-aligned bounding box.

    Args:
        obb (list): [x_center, y_center, width, height, angle_degrees]

    Returns:
        tuple: (x_min, y_min, x_max, y_max) of the axis-aligned bounding box
    """
    x_c, y_c, w, h, angle = obb
    angle_rad = np.deg2rad(angle)

    # Define OBB corners relative to center
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])

    # Rotate corners by angle
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ rotation_matrix.T

    # Translate corners to image coordinates
    rotated_corners[:, 0] += x_c
    rotated_corners[:, 1] += y_c

    # Compute AABB by finding min/max coordinates
    x_min = np.min(rotated_corners[:, 0])
    y_min = np.min(rotated_corners[:, 1])
    x_max = np.max(rotated_corners[:, 0])
    y_max = np.max(rotated_corners[:, 1])

    return int(x_min), int(y_min), int(x_max), int(y_max)


def get_obb_image_patch(image, obb, padding=0.1):
    x_c, y_c, w, h, angle = obb
    # Add padding to OBB dimensions
    w_padded = w * (1 + padding)
    h_padded = h * (1 + padding)

    # Compute AABB for the padded OBB
    padded_obb = [x_c, y_c, w_padded, h_padded, angle]
    x_min, y_min, x_max, y_max = obb_to_aabb(padded_obb)

    # Ensure AABB is within image bounds
    h_img, w_img = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w_img, x_max)
    y_max = min(h_img, y_max)

    # Crop the image patch
    patch = image[y_min:y_max, x_min:x_max]
    # Rotate the patch to align the object (remove OBB angle)
    patch_center = ((x_max - x_min) / 2, (y_max - y_min) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(patch_center, angle, 1.0)
    aligned_patch = cv2.warpAffine(patch, rotation_matrix, (x_max - x_min, y_max - y_min))

    # Compute new AABB in the aligned patch (object is now axis-aligned)
    new_w = w_padded
    new_h = h_padded
    new_x_c = (x_max - x_min) / 2
    new_y_c = (y_max - y_min) / 2
    aabb = [
        int(new_x_c - new_w / 2),
        int(new_y_c - new_h / 2),
        int(new_x_c + new_w / 2),
        int(new_y_c + new_h / 2)
    ]

    return aligned_patch, aabb, rotation_matrix


def run_yolo_pose():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load an official model
    # model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights
    # # Train the model
    # results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    # Access the results
    for result in results:
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)


def train_yolo_piglet():
    save_dir = os.path.join(ROOT_DIR, 'WTData/Results/250701_yolo')
    os.makedirs(save_dir, exist_ok=True)
    # https://docs.ultralytics.com/modes/train/#key-features-of-train-mode
    # https://docs.ultralytics.com/datasets/detect/
    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11.yaml')
    pretrained_model_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11s.pt')
    data_config_file = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2506/detection/data.yaml')
    # model = YOLO(model_config_file).load(pretrained_model_file)
    # # https://docs.ultralytics.com/usage/cfg/#modes
    # results = model.train(data=data_config_file, epochs=100, imgsz=640, project=save_dir)

    '''
    # # evaluation
    # model = YOLO("path/to/best.pt")
    # # Validate the model
    # metrics = model.val()
    # print(metrics.box.map)  # mAP50-95
    '''

    # tracking
    trackers = ['bytetrack', 'botsort', ] #]
    # video_file = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2506/train/ch12_20250109090000_000200.mp4')
    # video_file = os.path.join(ROOT_DIR, 'WTData/Piglet/clips/ch8_20250109093748_000000.mp4')
    video_file = os.path.join(ROOT_DIR, 'WTData/Piglet/clips/ch12_20250109090000_001700.mp4')
    model = YOLO("runs/detect/train/weights/yolo11n-best.pt")  # trained detector
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    # https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers
    for tracker in trackers:
        # box_results_file = os.path.join(save_dir, f'{video_name}_{tracker}_boxes.txt')
        # ids_results_file = os.path.join(save_dir, f'{video_name}_{tracker}_ids.txt')
        results_file = os.path.join(save_dir, f'{video_name}_{tracker}_results.txt')
        results_bin_file = os.path.join(save_dir, f'{video_name}_{tracker}_results.pt')
        results_json_file = os.path.join(save_dir, f'{video_name}_{tracker}_results.json')
        # consolidated torch tensors (gpu)
        results = model.track(video_file, show=False, tracker=f'{tracker}.yaml', save=False, project=save_dir)
        print(f'Finish tracking with {tracker}')
        # boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()
        # np.savetxt(box_results_file, boxes, fmt='%f')
        # np.savetxt(ids_results_file, track_ids, fmt='%d')
        # results = [result.cpu() for result in results]
        # results = torch.Tensor(results)
        # torch.save(results, results_bin_file)
        # print(f'Finish writing {results_bin_file}')
        # results = torch.load(results_bin_file)

        # '''
        results_json = {}
        # with open(results_file, 'w') as writer:
        #     writer.write("Frame\tID\tx\ty\tw\th\t\n")
        for i, result in enumerate(results):
            # sanity check to confirm meaningful predictions
            if len(result.boxes) == 0 or result.boxes is None:
                continue
            if result.boxes.id is None:
                continue
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            results_json[str(i)] = {}  # dictionary of detected objects in each frame
            for box, track_id in zip(boxes, track_ids):
                # writer.write(f'{i:d}\t{track_id:d}\t{box[0]:.3f}\t{box[1]:.3f}\t{box[2]:.3f}\t{box[3]:.3f}\n')
                # writer.flush()
                results_json[str(i)]['p'+str(track_id)] = box.tolist()
                # if i >= 3:
                #     break
        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))
        with open(results_json_file, 'w') as json_writer:
            json.dump(results_json, json_writer, indent=4)
        print(f'Finish writing {results_json_file}')
        # '''


def train_yolo11s_obb_detector():
    save_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb')
    os.makedirs(save_dir, exist_ok=True)
    # https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11
    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11-obb.yaml')
    # https://docs.ultralytics.com/tasks/obb/#visual-samples
    pretrained_model_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11s-obb.pt')
    data_config_file = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection/data.yaml')
    model = YOLO(model_config_file).load(pretrained_model_file)
    # # https://docs.ultralytics.com/usage/cfg/#modes
    # https://docs.ultralytics.com/modes/train/#key-features-of-train-mode
    # https://docs.ultralytics.com/modes/train/#train-settings
    results = model.train(data=data_config_file, batch=0.8, epochs=50, imgsz=640, project=save_dir)


def train_yolo8s_obb_detector():
    # https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes
    save_dir = os.path.join(ROOT_DIR, 'WTData/Results/250707_yolo8_obb')
    os.makedirs(save_dir, exist_ok=True)
    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolov8-obb.yaml')
    # https://docs.ultralytics.com/tasks/obb/#visual-samples
    # pretrained_model_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11s-obb.pt')
    data_config_file = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2506/obb_detection/data.yaml')
    model = YOLO(model_config_file).load("yolov8s-obb.pt")
    # # https://docs.ultralytics.com/usage/cfg/#modes
    # https://docs.ultralytics.com/modes/train/#key-features-of-train-mode
    # https://docs.ultralytics.com/modes/train/#train-settings
    results = model.train(data=data_config_file, batch=0.8, epochs=60, imgsz=640, project=save_dir)


def test_yolo11_abb_tracker():
    tracker = 'bytetrack'
    save_dir = os.path.join(ROOT_DIR, 'WTData/Results/250701_yolo')
    video_file = os.path.join(ROOT_DIR, 'WTData/Piglet/clips/ch12_20250109090000_001700.mp4')
    gt_csv_path = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2506/val/ch12_20250109090000_001700_mot.txt')
    detector_model_file = "runs/detect/train/weights/yolo11n-best.pt"

    # detector evaluation
    model = YOLO(detector_model_file)
    # Validate the model
    metrics = model.val()
    print(metrics.box.map)  # mAP50-95

    results = model.track(video_file, tracker=f'{tracker}.yaml', save=False, save_txt=True,
                          project=save_dir, name="eval_run", show=False, verbose=True)
    # tracker evaluation
    if gt_csv_path is not None:
        print("Loading tracking predictions for evaluation...")

        # Load predictions
        # pred_files = list(Path(f"{save_dir}/eval_run/labels").glob("*.txt"))
        acc = mm.MOTAccumulator(auto_id=True)

        # for pred_file in pred_files:
        for pred_file in sorted(glob.glob(f"{save_dir}/eval_run/labels*.txt")):
            frame_id = int(pred_file.stem.split("_")[-1])
            preds = pd.read_csv(pred_file, header=None, sep=" ")
            # Expected columns: [class_id, track_id, x_center, y_center, width, height, conf]
            preds.columns = ["class_id", "track_id", "xc", "yc", "w", "h", "conf"]
            # yolo output format (percentage):
            # [class_id, x_center, y_center, width, height, track_id]

            # Load corresponding ground truth for this frame
            # This assumes your gt_csv_path contains columns:
            # [frame, id, x, y, w, h, class]
            # MOT format (pixel):
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            gt_df = pd.read_csv(gt_csv_path)
            gt_df.columns = ["frame", "track_id", "x", "y", "w", "h", "conf", "x3", "y3", "z3"]
            gt_frame = gt_df[gt_df["frame"] == frame_id]

            # Convert boxes to [x1, y1, x2, y2]
            def mot_xywh_to_xyxy(df):
                df["x1"] = df["x"] - df["w"] / 2
                df["y1"] = df["y"] - df["h"] / 2
                df["x2"] = df["x"] + df["w"] / 2
                df["y2"] = df["y"] + df["h"] / 2
                return df

            preds = xywh_to_xyxy(preds)
            gt_frame = xywh_to_xyxy(gt_frame)

            # Build distance matrix using IoU
            def iou_matrix(gt_boxes, pred_boxes):
                iou = mm.distances.iou_matrix(
                    gt_boxes[["x1", "y1", "x2", "y2"]].values,
                    pred_boxes[["x1", "y1", "x2", "y2"]].values,
                    max_iou=0.5
                )
                return iou

            if not gt_frame.empty and not preds.empty:
                dists = iou_matrix(gt_frame, preds)
                acc.update(
                    gt_frame["id"].values,
                    preds["track_id"].values,
                    dists
                )
            else:
                acc.update(
                    gt_frame["id"].values if not gt_frame.empty else [],
                    preds["track_id"].values if not preds.empty else [],
                    np.empty((len(gt_frame), len(preds)))
                )

        mh = mm.metrics.create()
        summary = mh.compute(
            acc,
            metrics=mm.metrics.motchallenge_metrics,
            name="bytetrack_eval"
        )
        print(mm.io.render_summary(summary, formatters=mh.formatters))
    else:
        print("No ground truth provided. Skipping quantitative evaluation.")


def test_yolo_obb_detector():
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection')
    visuals_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train')
    pretrained_model_file = os.path.join(visuals_dir, 'weights/250722-yolo11s-obb-best.pt')
    data_config_file = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection/data.yaml')
    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11-obb.yaml')
    model = YOLO(model_config_file).load(pretrained_model_file)
    test_image_dir = os.path.join(dataset_dir, 'images/val')
    test_gt_dir = os.path.join(dataset_dir, 'labels/val')
    for file in sorted(glob.glob(os.path.join(test_image_dir, '*.jpg'))):
        file_name = os.path.basename(file)
        file_name = os.path.splitext(file_name)[0]
        results = model(file)
        # locations = results[0].obb.xyxyxyxy.cpu().numpy()  # N * 4 * 2

        # names = dict(results[0].names)
        locations = results[0].obb.xywhr.cpu().numpy()
        image_bgr = cv2.imread(file)
        for loc in locations:
            x_center, y_center, width, height, angle = loc#.xywhr[0]
            x_center, y_center = int(x_center), int(y_center)
            angle = np.degrees(angle)
            box = cv2.boxPoints(((x_center, y_center), (width, height), angle))
            box = box.astype(np.int32)
            cv2.drawContours(image_bgr, [box], -1, color=(255,255,255), thickness=2)
        result_img_file = os.path.join(visuals_dir, f'yolo11n_obb_{file_name}.png')
        image_bgr = image_bgr[:, 500:2050, :]
        # cv2.putText(image_bgr, f'frame{n}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
        #             color=(255, 255, 255), thickness=3)
        cv2.imwrite(result_img_file, image_bgr)
        print(f'Finish writing {result_img_file}')
        # exit(0)


def run_yolo_obb_on_seq_save_result_csv():
    clips = {
        'train': ['ch2_20250109090000_000200',
                  'ch6_20250109093801_000500',
                  'ch8_20250109093748_000000',
                  'ch11_20250109094853_000500',
                  'ch12_20250109090000_000200',
                  'ch12_20250109090000_003000',
                  ],
        'val': ['ch1_20250109090000_001900',
                'ch12_20250109090000_001700',
                ]
    }
    pen_sides = {
        'ch1': [730, 2050],
        'ch2': [650],
        'ch6': [730, 2020],
        'ch8': [550, 1870],
        'ch11': [570, 2090],
        'ch12': [570, 1870],
    }
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection')

    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11-obb.yaml')
    model_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train')
    pretrained_model_file = os.path.join(model_dir, 'weights/250722-yolo11s-obb-best.pt')
    model = YOLO(model_config_file).load(pretrained_model_file)
    for split, clips in clips.items():
        for clip_name in clips:
            cam = clip_name.split('_')[0]
            # result_csv_file = os.path.join(model_dir, f'yolo11n-obb-{clip_name}-predictions.txt')
            result_csv_file = os.path.join(model_dir, f'yolo11n-obb-{clip_name}-predictions-filtered.txt')
            test_image_dir = os.path.join(dataset_dir, f'images/{split}')
            fw = open(result_csv_file, 'w')
            writer = csv.writer(fw, delimiter=',', quotechar='"')
            print(f'Running {pretrained_model_file} on {clip_name}')
            for file in sorted(glob.glob(os.path.join(test_image_dir, f'{clip_name}_*.jpg'))):
                file_name = os.path.basename(file)
                file_name = os.path.splitext(file_name)[0]
                frame_id = int(file_name.split('_')[-1])
                results = model(file)
                locations = results[0].obb.xyxyxyxy.cpu().numpy()
                conf_scores = results[0].obb.conf.cpu().numpy()
                locations = locations.astype(np.int32)
                for loc, conf in zip(locations, conf_scores):
                    x1, y1, x2, y2 = loc[0, 0], loc[0, 1], loc[1, 0], loc[1, 1]
                    x3, y3, x4, y4 = loc[2, 0], loc[2, 1], loc[3, 0], loc[3, 1]
                    # Jian: add post-processing to eliminate out-of-pen detections
                    # question: how to determine out-of-pen?
                    valid = False
                    if cam == 'ch2':  # half pen
                        ymin = np.min([y1, y2, y3, y4])
                        if ymin > pen_sides[cam][0]:
                            valid = True
                    else:
                        xmin = np.min([x1, x2, x3, x4])
                        xmax = np.max([x1, x2, x3, x4])
                        if xmin < pen_sides[cam][1] and xmax > pen_sides[cam][0]:
                            valid = True
                    if valid:
                        writer.writerow([frame_id, conf, x1, y1, x2, y2, x3, y3, x4, y4])
                if frame_id % 300 == 0:
                    print(f'Image predicted: {file_name}')
            fw.close()


def filter_saved_yolo_obb_results():
    clips = {
        'train': ['ch2_20250109090000_000200',
                  'ch6_20250109093801_000500',
                  'ch8_20250109093748_000000',
                  'ch11_20250109094853_000500',
                  'ch12_20250109090000_000200',
                  'ch12_20250109090000_003000',
                  ],
        'val': ['ch1_20250109090000_001900',
                'ch12_20250109090000_001700',
                ]
    }
    pen_sides = {
        'ch1': [730, 2050],
        'ch2': [650],
        'ch6': [730, 2020],
        'ch8': [550, 1870],
        'ch11': [570, 2090],
        'ch12': [570, 1870],
    }
    model_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train')
    for split, clips in clips.items():
        for clip_name in clips:
            cam = clip_name.split('_')[0]
            in_result_csv_file = os.path.join(model_dir, f'yolo11n-obb-{clip_name}-predictions.txt')
            out_result_csv_file = os.path.join(model_dir, f'yolo11n-obb-{clip_name}-predictions-filtered.txt')
            predictions = np.loadtxt(in_result_csv_file, delimiter=',')
            # filtered_predictions = []
            fw = open(out_result_csv_file, 'w')
            writer = csv.writer(fw, delimiter=',', quotechar='"')
            for row in predictions:
                int_row = row[2:].astype(np.int32)
                # x1, y1, x2, y2 = row[2], row[3], row[4], row[5]
                # x3, y3, x4, y4 = row[6], row[7], row[8], row[9]
                x1, y1, x2, y2 = int_row[0], int_row[1], int_row[2], int_row[3]
                x3, y3, x4, y4 = int_row[4], int_row[5], int_row[6], int_row[7]
                valid = False
                if cam == 'ch2':  # half pen
                    ymin = np.min([y1, y2, y3, y4])
                    if ymin > pen_sides[cam][0]:
                        valid = True
                else:
                    xmin = np.min([x1, x2, x3, x4])
                    xmax = np.max([x1, x2, x3, x4])
                    if xmin < pen_sides[cam][1] and xmax > pen_sides[cam][0]:
                        valid = True
                if valid:
                    # filtered_predictions.append(row)
                    writer.writerow([int(row[0]), f"{row[1]:.3f}", x1, y1, x2, y2,
                                     x3, y3, x4, y4])
            # np.savetxt(out_result_csv_file, filtered_predictions, delimiter=',', fmt='%d')
            fw.close()


def calculate_yolo_obb_detection_metrics():
    clips = {
        'train': ['ch2_20250109090000_000200',
                  'ch6_20250109093801_000500',
                  'ch8_20250109093748_000000',
                  'ch11_20250109094853_000500',
                  'ch12_20250109090000_000200',
                  'ch12_20250109090000_003000',
                  ],
        'val': ['ch1_20250109090000_001900',
                'ch12_20250109090000_001700',
                ]
    }
    dataset_dir = os.path.join(ROOT_DIR, 'WTData', 'Dataset', 'Piglet2507')
    model_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train')
    # # txt writer
    # evaluate_result_file = os.path.join(model_dir, 'evaluate_result.txt')
    # fw = open(evaluate_result_file, 'w')
    # io_utils.log_with_print(fw,"Oriented Bounding Box Detection Performance")
    # io_utils.log_with_print(fw,"Split\t\tSequence\t\tAP\tPrecision\tRecall")
    # csv writer
    # evaluate_result_file = os.path.join(model_dir, 'evaluate_result.csv')
    evaluate_result_file = os.path.join(model_dir, 'evaluate_result_filtered.csv')
    # # remove previous csv file
    # os.chdir(model_dir)
    # command = "rm evaluate_result_filtered.csv"
    # os.system(command)

    fw = open(evaluate_result_file, 'w')
    writer = csv.writer(fw, delimiter=',', quotechar='"')
    writer.writerow(['Split', 'Sequence', 'AP', 'Precision', 'Recall'])
    for split, clips in clips.items():
        for clip_name in clips:
            gt_path = os.path.join(dataset_dir, f'{clip_name}_obb_tracking.txt')
            # det_path = os.path.join(model_dir, f'yolo11n-obb-{clip_name}-predictions.txt')
            det_path = os.path.join(model_dir, f'yolo11n-obb-{clip_name}-predictions-filtered.txt')
            gt_data = eval_utils.read_obb_gt_file(gt_path)
            det_data = eval_utils.read_obb_det_file(det_path)
            print(f"Evaluating {clip_name}")
            # mAP, precision (how many correct in predicted), recall (how many predicted in true)
            results = eval_utils.evaluate_obb_det(gt_data, det_data, iou_thresh=0.5)
            # io_utils.log_with_print(fw, f"{split}\t{clip_name}\t{results['AP']:.3f}\t{results['Precision']:.3f}\t{results['Recall']:.3f}")
            writer.writerow([split, clip_name, f"{results['AP']:.3f}", f"{results['Precision']:.3f}", f"{results['Recall']:.3f}"])
    fw.close()


def run_piglet_detector_save_result_into_ls_json_trial():
    image_name = "ch6_20250109093801_0001.jpg"
    image_width, image_height = 2688, 1520
    test_image_dir = os.path.join(ROOT_DIR, f'WTData/Piglet/Raw/obb_images')
    dataset_dir = os.path.join(ROOT_DIR, f'WTData/Dataset/Piglet2508')
    project_without_anno_json = os.path.join(dataset_dir, 'project-24-at-2025-07-25-14-38-a6119ff7.json')
    prediction_json = os.path.join(dataset_dir, 'project-24-predictions.json')
    image_file = os.path.join(test_image_dir, image_name)

    # trained detector model
    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11-obb.yaml')
    model_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train')
    pretrained_model_file = os.path.join(model_dir, 'weights/250722-yolo11s-obb-best.pt')
    model = YOLO(model_config_file).load(pretrained_model_file)

    # prediction json to write
    image_folder = "Documents/Work/QUBRF/Piglets/Data/MSUclips/obb_labelling"
    prefix = f"/data/local-files/?d={image_folder}/"
    json_contents = []  # list of all tasks
    f = open(project_without_anno_json, 'r')
    contents = json.load(f)
    f.close()
    for tasks in contents:
        image_task = {}
        image_task["id"] = tasks["id"]  # parallel to annotations, file_upload, created_at, predictions[]
        image_task["data"] = {}  # parallel to id, annotations
        image_task["data"]["image"] = tasks["data"]["image"]  # single image in one task
        # print(image_file)
        # print(tasks["data"]["image"])
        image_task["predictions"] = []  # list of ?
        entry = {}  # include result & id, inside annotations, paralell to prediction{}
        entry["model_version"] = 'yolo11n-obb'
        entry["score"] = 0.6
        entry["result"] = []  # copy all result into prediction
        # for annotations in tasks["annotations"]:
        #     for anno_result in annotations["result"]:  # list of objects
        #         result = {}

        results = model(image_file)
        locations = results[0].obb.xywhr.cpu().numpy()
        conf_scores = results[0].obb.conf.cpu().numpy()
        box_count = 0
        for loc, conf in zip(locations, conf_scores):
            conf = float(conf)
            if conf < 0.5:
                continue  # discard boxes with a confidence score lower than 0.5
            # rotation angle in radians,
            # measures the angle from the horizontal axis to the bounding box's major axis in a counter-clockwise direction
            # https://docs.ultralytics.com/datasets/obb/#yolo-obb-format
            x_center, y_center, width, height, angle = loc#.xywhr[0]
            xtl = x_center - width / 2
            ytl = y_center - height / 2
            # x_center, y_center = int(x_center), int(y_center)
            angle = np.degrees(angle)
            # yolo output to LS format
            obb = {}
            obb["original_width"] = image_width
            obb["original_height"] = image_height
            obb["image_rotation"] = 0
            # obb["id"] = "fq"
            obb["from_name"] = "label"
            obb["to_name"] = "image"
            obb["type"] = "rectanglelabels"
            obb["origin"] = "manual"
            obb["value"] = {}
            obb["value"]["x"] = float(xtl / image_width * 100.0)
            obb["value"]["y"] = float(ytl / image_height * 100.0)
            obb["value"]["width"] = float(width / image_width * 100.0)
            obb["value"]["height"] = float(height / image_height * 100.0)
            obb["value"]["rotation"] = float(angle)
            obb["value"]["rectanglelabels"] = ["piglet"]
            entry["result"].append(obb)
            box_count += 1
        image_task["predictions"].append(entry)
        json_contents.append(image_task)
        print(f'Detected {box_count} boxes (of {len(conf_scores)})')
        break
    # save predictions
    with open(prediction_json, 'w') as fw:
        json.dump(json_contents, fw, indent=4)
    print(f'Finish saving json at {prediction_json}')


def run_piglet_detector_save_result_into_ls_json():
    model_name = 'yolo11n-obb-0805'
    # model_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train')
    # pretrained_model_file = os.path.join(model_dir, 'weights/250722-yolo11s-obb-best.pt')
    pretrained_model_file = os.path.join(ROOT_DIR, 'WTData/Results/250805_yolo11n_obb/train2/weights/250805-best-yolo11n-obb.pt')
    yolo_conf_threshold = 0.5
    image_width, image_height = 2688, 1520
    test_image_dir = os.path.join(ROOT_DIR, f'WTData/Piglet/Raw/obb_images')
    dataset_dir = os.path.join(ROOT_DIR, f'WTData/Dataset/Piglet2508/batch0805')
    # image_prediction_template_json = os.path.join(dataset_dir, 'image_task_empty_predictions.json')
    # image_prediction_output_json = os.path.join(dataset_dir, 'image_task_with_predictions.json')
    image_prediction_template_json = os.path.join(dataset_dir, 'image2729_task_empty_predictions.json')
    remove_sample_list = np.loadtxt(os.path.join(dataset_dir, 'consolidated_labels_list.txt'), dtype='str')
    # trained detector model
    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11-obb.yaml')
    model = YOLO(model_config_file).load(pretrained_model_file)

    json_contents = []  # list of all tasks
    f = open(image_prediction_template_json, 'r')
    contents = json.load(f)
    f.close()
    remove_count = 0
    active_task_count = 0
    for tasks in contents:
        image_task = {}
        image_task["id"] = tasks["id"]  # parallel to annotations, file_upload, created_at, predictions[]
        image_task["data"] = {}  # parallel to id, annotations
        image_task["data"]["image"] = tasks["data"]["image"]
        image_task["predictions"] = []
        entry = {}  # include result & id, inside annotations, paralell to prediction{}
        entry["model_version"] = model_name
        entry["score"] = 0.6
        entry["result"] = []  # copy all result into prediction

        image_name = os.path.basename(tasks["data"]["image"])
        # remove images that have already been manually corrected
        sample_name = os.path.splitext(image_name)[0]
        if sample_name in remove_sample_list:
            remove_count += 1
            continue

        # # for debugging -------------------------------------------
        # if image_name != 'ch12_20250109090000_0108.jpg':
        #     continue
        # yolo_results_file = os.path.join(dataset_dir, f'yolo_results.txt')
        # # for debugging -------------------------------------------
        image_file = os.path.join(test_image_dir, image_name)
        results = model(image_file)
        # locations = results[0].obb.xywhr.cpu().numpy()
        locations = results[0].obb.xyxyxyxy.cpu().numpy()
        conf_scores = results[0].obb.conf.cpu().numpy()
        box_count = 0
        # yolo_obbs = []
        for loc, conf in zip(locations, conf_scores):
            conf = float(conf)
            if conf < yolo_conf_threshold:
                continue  # discard boxes with a confidence score lower than threshold
            x, y, w, h, angle = ls_io_utils.yolo_obb_to_labelstudio_obb(loc, image_width, image_height)
            # yolo_obbs.append(loc.flatten().tolist())
            obb = {}
            obb["original_width"] = image_width
            obb["original_height"] = image_height
            obb["image_rotation"] = 0
            obb["from_name"] = "label"
            obb["to_name"] = "image"
            obb["type"] = "rectanglelabels"
            obb["origin"] = "manual"
            obb["value"] = {}
            obb["value"]["x"] = x
            obb["value"]["y"] = y
            obb["value"]["width"] = w
            obb["value"]["height"] = h
            obb["value"]["rotation"] = angle
            obb["value"]["rectanglelabels"] = ["piglet"]
            entry["result"].append(obb)
            box_count += 1
        image_task["predictions"].append(entry)
        json_contents.append(image_task)
        active_task_count += 1
        print(f'Detected {box_count} boxes (of {len(conf_scores)})')
        # np.savetxt(yolo_results_file, yolo_obbs, fmt='%.6f', delimiter='\t', newline='\n')
    # save predictions
    image_prediction_output_json = os.path.join(dataset_dir, f'image{active_task_count}_task_with_predictions.json')
    with open(image_prediction_output_json, 'w') as fw:
        json.dump(json_contents, fw, indent=4)
    print(f'Finish saving json with {active_task_count} active tasks ({remove_count} removed)')


def finetune_yolo11n_obb_detector():
    save_dir = os.path.join(ROOT_DIR, 'WTData/Results/250805_yolo11n_obb')
    os.makedirs(save_dir, exist_ok=True)
    pretrained_model_file = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train/weights/250722-yolo11s-obb-best.pt')
    data_config_file = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/batch0805/data.yaml')
    model_config_file = os.path.join(ROOT_DIR, 'WTData/Code/yolo_configs/yolo11-obb.yaml')
    model = YOLO(model_config_file).load(pretrained_model_file)
    results = model.train(data=data_config_file, batch=0.8, epochs=50, imgsz=640, project=save_dir,
                          cache=True,  # Enables caching of dataset images in memory
                          )


if __name__ == '__main__':
    # run_yolo_obb()
    # train_yolo_piglet()
    # train_yolo11s_obb_detector()
    # train_yolo8s_obb_detector()
    # test_yolo11_abb_tracker()
    # test_yolo_obb_detector()
    # run_yolo_obb_on_seq_save_result_csv()
    # calculate_yolo_obb_detection_metrics()
    # filter_saved_yolo_obb_results()
    run_piglet_detector_save_result_into_ls_json()
    # finetune_yolo11n_obb_detector()