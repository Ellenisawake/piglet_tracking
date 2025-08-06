import csv
import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict
import glob
import os
from config import ROOT_DIR



def read_obb_gt_file(gt_path):
    gt_data = defaultdict(list)
    # format: ['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    with open(gt_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame_id = int(row[0])
            track_id = int(row[1])
            points = list(map(float, row[2:]))
            poly = Polygon([(points[i], points[i+1]) for i in range(0,8,2)])
            gt_data[frame_id].append({'track_id': track_id, 'polygon': poly})
    return gt_data


def read_obb_det_file(det_path):
    det_data = defaultdict(list)
    # format: ['frame_id', 'conf', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    with open(det_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame_id = int(row[0])
            conf = float(row[1])
            points = list(map(float, row[2:]))
            poly = Polygon([(points[i], points[i+1]) for i in range(0,8,2)])
            det_data[frame_id].append({'conf': conf, 'polygon': poly})
    return det_data


def compute_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0.0
    return inter / union


def evaluate_obb_det(gt_data, det_data, iou_thresh=0.5):
    all_detections = []
    all_gts = 0
    for frame_id in gt_data.keys():
        gt_polys = gt_data[frame_id]
        det_polys = det_data.get(frame_id, [])

        matched_gt = set()
        detections = sorted(det_polys, key=lambda x: -x['conf'])
        frame_tp = 0
        frame_fp = 0

        for det in detections:
            matched = False
            for idx, gt in enumerate(gt_polys):
                if idx in matched_gt:
                    continue
                iou = compute_iou(det['polygon'], gt['polygon'])
                if iou >= iou_thresh:
                    matched = True
                    matched_gt.add(idx)
                    break
            if matched:
                frame_tp += 1
                all_detections.append((det['conf'], 1))
            else:
                frame_fp += 1
                all_detections.append((det['conf'], 0))
        all_gts += len(gt_polys)

    all_detections.sort(reverse=True, key=lambda x: x[0])
    tp_cumsum = np.cumsum([d[1] for d in all_detections])
    fp_cumsum = np.cumsum([1 - d[1] for d in all_detections])
    # high recall, less miss
    recalls = tp_cumsum / all_gts if all_gts > 0 else np.zeros(len(tp_cumsum))
    # high precision, less false alarm
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    ap = compute_ap(recalls, precisions)

    return {
        'AP': ap,
        'Precision': precisions[-1] if len(precisions) > 0 else 0.0,
        'Recall': recalls[-1] if len(recalls) > 0 else 0.0
    }


def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


