import os
import cv2
import glob
import json
import numpy as np
from config import ROOT_DIR, BOX_COLORS


def draw_yolo_track_results():
    num_val = 200
    tracker = 'botsort'  # 'bytetrack'
    video_name = 'ch8_20250109093748_000000'  # 'ch12_20250109090000_000200'  # 'ch12_20250109090000_001700'
    image_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2506/detection/images/val')
    result_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250701_yolo')
    vis_dir = os.path.join(result_dir, f'{video_name}_{tracker}')
    os.makedirs(vis_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f'{video_name}_{tracker}_results.json')
    with open(result_file, 'r') as reader:
        results = json.load(reader)
        for frame_str, detections in results.items():
            frame_number = int(frame_str)
            image_file = os.path.join(image_dir, f'{video_name}_{frame_number:04d}.jpg')
            if not os.path.exists(image_file):
                continue
            if len(detections) == 0:
                continue
            if frame_number >= num_val:
                break
            vis_file = os.path.join(vis_dir, f'{video_name}_{frame_number:04d}_{tracker}.jpg')
            detection_count = 0
            image = cv2.imread(image_file)
            for pid in detections.keys():
                xc, yc, w, h = detections[pid]
                color = BOX_COLORS[pid]
                x0, x1 = int(xc - w / 2), int(xc + w / 2)
                y0, y1 = int(yc - h / 2), int(yc + h / 2)
                xc, yc = int(xc), int(yc)
                cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=2)
                cv2.circle(image, (xc, yc), radius=6, color=color, thickness=-1)
                cv2.putText(image, str(pid), (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=color, thickness=3)
                detection_count += 1
            cv2.putText(image, f'frame {frame_number}', (50, 1500), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=(255, 255, 255), thickness=3)
            cv2.imwrite(vis_file, image)
            print(f'Finish plotting {detection_count} detections for frame {frame_number}')


def draw_yolo_obb_label_on_image(image_dir, label_dir, clip_name, fid, img_w, img_h):
    image_file = os.path.join(image_dir, f'{clip_name}_{fid:04d}.jpg')
    label_file = os.path.join(label_dir, f'{clip_name}_{fid:04d}.txt')
    if not os.path.exists(image_file):
        print(f'Skipping because image not found: {image_file}')
        return None
    image = cv2.imread(image_file)
    cam = clip_name.split('_')[0]
    if os.path.exists(label_file):
        # class_index x1 y1 x2 y2 x3 y3 x4 y4 (normalized between 0 and 1)
        labels = np.loadtxt(label_file, delimiter='\t', dtype=np.float32)
        for i, label in enumerate(labels):
            if len(label) < 2:
                print(f'Incomplete labels in {label_file} line {i}')
                continue
            x1 = int(label[1] * img_w)
            x2 = int(label[3] * img_w)
            x3 = int(label[5] * img_w)
            x4 = int(label[7] * img_w)
            y1 = int(label[2] * img_h)
            y2 = int(label[4] * img_h)
            y3 = int(label[6] * img_h)
            y4 = int(label[8] * img_h)
            # if i > 17:  # assume piglet count <=18
            #     print(f'Number of objects in {split} {sample_name}: {len(labels)}')
            #     exit(0)
            color = BOX_COLORS['p' + str(i + 1)]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness=2)
            cv2.line(image, (x2, y2), (x3, y3), color, thickness=2)
            cv2.line(image, (x3, y3), (x4, y4), color, thickness=2)
            cv2.line(image, (x1, y1), (x4, y4), color, thickness=2)
            cv2.putText(image, f'{cam} frame {fid}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 255, 255), thickness=2)
            cv2.putText(image, f'{cam} frame {fid}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=color, thickness=1)
    else:
        print(f'label_file not found: {label_file}')
    cv2.putText(image, f'{cam} frame {fid}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=(255, 255, 255), thickness=3)
    return image


def visualise_tracklab_predictions():
    video_name = 'ch12_20250109090000_003000'
    split = 'train'
    result_dir = os.path.join(ROOT_DIR, f'WTData/Results/250731_tracklab/tracklab/2025-08-05/23-22-54/eval/pred/Piglet-{split}/tracklab')
    prediction_file = os.path.join(result_dir, f'{video_name}.txt')
    dataset_dir = os.path.join(ROOT_DIR, f'WTData/Dataset/Piglet2507/tracklab')
    image_dir = os.path.join(dataset_dir, f'{split}/{video_name}/img1')
    vis_save_dir = os.path.join(result_dir, 'vis_' + video_name)
    os.makedirs(vis_save_dir, exist_ok=True)
    # frame, track_id, left, top, width, height, confidence, -1, -1, -1
    predictions = np.loadtxt(prediction_file, delimiter=',')
    ids = np.unique(predictions[:, 1]).astype(np.int32)
    id_remaps = {}
    new_id = 1
    for id in ids:
        if id not in id_remaps.keys():
            id_remaps[id] = new_id
            new_id += 1
    print(id_remaps)
    frame_count = 0
    for file in sorted(glob.glob(os.path.join(image_dir, f'{video_name}_*.jpg'))):
        image_name = os.path.splitext(os.path.basename(file))[0]
        frame_index = int(image_name.split('_')[-1])
        cam = image_name.split('_')[0]
        frame_boxes = predictions[predictions[:, 0] == frame_index]
        image = cv2.imread(file)
        for box in frame_boxes:
            pid = id_remaps[int(box[1])]
            color = BOX_COLORS['p' + str(pid)]
            x0, y0 = int(box[2]), int(box[3])
            x1, y1 = int(x0 + box[4]), int(y0 + box[5])
            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=2)
            cv2.putText(image, 'p' + str(pid), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=color, thickness=2)
        cv2.putText(image, f'{cam} frame {frame_index}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    color=(255, 255, 255), thickness=3)
        cv2.imwrite(os.path.join(vis_save_dir, f'vis_{image_name}.jpg'), image)
        frame_count += 1
        if frame_count == 3:
            exit(0)
    print(f'Finish visualising {video_name} with {frame_count} frames')



if __name__ == '__main__':
    # draw_yolo_track_results()
    visualise_tracklab_predictions()