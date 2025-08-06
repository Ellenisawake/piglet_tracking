import os
import cv2
import glob
import json
import copy
import numpy as np

from config import ROOT_DIR
import data_utils
import ls_io_utils


def check_image_obb_json():
    in_json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudiocheck/2508_obb_det')
    # bulk_json_file = os.path.join(in_json_dir, 'project-14-at-2025-07-24-10-28-3c66671a.json')
    # save_json_file = os.path.join(in_json_dir, 'project-14-restructured.json')
    bulk_json_file = os.path.join(in_json_dir, 'project-2-at-2025-08-01-10-27-656c2891.json')
    save_json_file = os.path.join(in_json_dir, 'project-2-restructured.json')
    ls_io_utils.write_ls_export_json_into_structured(bulk_json_file, save_json_file)


def extract_frames_for_obb_labelling():
    extract_interval = 590
    # extract_channels = ['ch3', 'ch6', 'ch8', 'ch9', 'ch11', 'ch12', 'ch14', 'ch16', 'ch17']
    data_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/Raw')
    det_dir = os.path.join(data_dir, 'obb_images')
    os.makedirs(det_dir, exist_ok=True)
    os.chdir(data_dir)
    for file in sorted(glob.glob(os.path.join(data_dir, 'piglet_nvr_*.mp4'))):
        basename = os.path.basename(file)
        tags = basename.split('_')
        channel = tags[2]
        # if channel not in extract_channels:
        #     continue
        timestamp = tags[4]
        clip_name = channel + '_' + timestamp
        command = f'ffmpeg -i {basename} -vf "select=not(mod(n\,{extract_interval})),setpts=N/FRAME_RATE/TB" -vsync vfr obb_images/{clip_name}_%04d.jpg'
        os.system(command)
        print(f'Clip extracted: {basename}')
        # exit(0)


def remove_tracking_frames_from_obb_labelling_images():
    det_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/Raw/obb_images')
    os.chdir(det_dir)
    for file in sorted(glob.glob(os.path.join(det_dir, 'tracking_*.jpg'))):
        basename = os.path.basename(file)
        command = f'rm {basename}'
        os.system(command)
        print(f'Tracking frame removed: {basename}')


def check_soworientation_dataset():
    image_dir = os.path.join(ROOT_DIR, "WTData/piglet_open_datasets/SowOrientation/SowOrientation_Dataset/images")
    label_dir = os.path.join(ROOT_DIR, "WTData/piglet_open_datasets/SowOrientation/SowOrientation_Dataset/labels")
    image_file = os.path.join(image_dir, "NoBirth_DG12_A202_Kamera2-20220625-174350-1656171830_600_1200_frame11000.png")
    label_file = os.path.join(label_dir, "NoBirth_DG12_A202_Kamera2-20220625-174350-1656171830_600_1200_frame11000.txt")
    image_to_save = os.path.join(ROOT_DIR, "WTData/piglet_open_datasets/SowOrientation/SowOrientation_NoBirth_DG12_A202_Kamera2-20220625-174350-1656171830_600_1200_frame11000.png")
    image = cv2.imread(image_file)
    img_h, img_w = image.shape[:2]  # 640*640
    labels = np.loadtxt(label_file, dtype=np.float32)
    for row in labels:
        obj_id = str(int(row[0]))
        x = int(row[1] * img_w)
        y = int(row[2] * img_h)
        w = int(row[3] * img_w)
        h = int(row[4] * img_h)
        xtl = x - w / 2
        ytl = x - w / 2
        cv2.rectangle(image, (x,y),(x+w, y+h), (0,255,0), 2)
        cv2.putText(image, obj_id, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=3)
    cv2.imwrite(image_to_save, image)


def find_detection_post_processing_pen_areas():
    import io_utils
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
    evaluate_result_file = os.path.join(dataset_dir, 'pen_valid_condition.txt')
    fw = open(evaluate_result_file, 'w')
    io_utils.log_with_print(fw,"Pen valid condition")
    for split, clips in clips.items():
        for clip_name in clips:
            cam = clip_name.split('_')[0]
            sample_csv_file = os.path.join(dataset_dir, f'{clip_name}_obb_tracking.txt')
            # frame, obj_id, x1 y1 x2 y2 x3 y3 x4 y4 (in pixels)
            all_tracks = np.loadtxt(sample_csv_file, delimiter=',', dtype=np.int32)
            left_bounds = min(np.min(all_tracks[:, 2]), np.min(all_tracks[:, 4]), np.min(all_tracks[:, 6]), np.min(all_tracks[:, 8]))
            right_bounds = max(np.max(all_tracks[:, 2]), np.max(all_tracks[:, 4]), np.max(all_tracks[:, 6]), np.max(all_tracks[:, 8]))
            top_bounds = min(np.min(all_tracks[:, 3]), np.min(all_tracks[:, 5]), np.min(all_tracks[:, 7]), np.min(all_tracks[:, 9]))
            bottom_bounds = max(np.max(all_tracks[:, 3]), np.max(all_tracks[:, 5]), np.max(all_tracks[:, 7]), np.max(all_tracks[:, 9]))
            if cam == 'ch2':  # pen-of-interest is lower half of frame
                valid_condition = f"y > {top_bounds} (top_bounds)"
            else:
                valid_condition = f"{left_bounds} (left_bounds) < x < {right_bounds} (right_bounds)"
            io_utils.log_with_print(fw, clip_name)
            io_utils.log_with_print(fw,f"valid_condition: {valid_condition}")


def ls_json_to_yolo_obb0722():
    spec_frame = -1  # 1  #
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
    img_w, img_h = 2688.0, 1520.0
    json_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'full_annotation')
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507')
    for split, clips in clips.items():
        image_dir = os.path.join(dataset_dir, f'obb_detection/images/{split}')
        obb_label_dir = os.path.join(dataset_dir, f'obb_detection/labels/{split}')
        os.makedirs(obb_label_dir, exist_ok=True)
        for clip_name in clips:
            anno_json_file = os.path.join(json_dir, f'{clip_name}_with_interpolations.json')
            if not os.path.exists(anno_json_file):
                print(f'Skipping because anno JSON not found: {anno_json_file}')
                continue
            label_file_count = ls_io_utils.write_ls_video_export_json_into_yolo_obb(anno_json_file, obb_label_dir,
                                                                                    image_dir, img_w, img_h, spec_frame)
            print(f'Label files written for {clip_name}: {label_file_count}')


def ls_json_to_mot_csv250722():
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
    img_w, img_h = 2688, 1520
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'full_annotation')
    save_dir = os.path.join(ROOT_DIR, 'WTData', 'Dataset', 'Piglet2507')
    for split, clips in clips.items():
        for clip_name in clips:
            anno_json_file = os.path.join(data_dir, f'{clip_name}_with_interpolations.json')
            # out_csv_file = os.path.join(save_dir, f'{clip_name}_mot.txt')
            # ls_io_utils.write_ls_video_obb_export_json_into_mot_abb_csv(anno_json_file, out_csv_file, img_w, img_h)
            # print(f'Finished writing MOT GT for {clip_name}')
            out_csv_file = os.path.join(save_dir, f'{clip_name}_obb_tracking.txt')
            ls_io_utils.write_ls_json_obb_to_obb_tracking_gt_csv(anno_json_file, out_csv_file, img_w, img_h)
            print(f'Finished writing OBB Tracking GT for {clip_name}')


def check_video_length():
    fps = 30.0
    data_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/Raw')
    os.chdir(data_dir)
    record_file = os.path.join(data_dir, 'records.json')
    record_json = {}
    total_frames_count = 0
    for file in sorted(glob.glob(os.path.join(data_dir, 'piglet_nvr_*.mp4'))):
        basename = os.path.basename(file)
        tags = basename.split('_')
        channel = tags[2]
        timestamp = tags[4]
        clip_name = channel + '_' + timestamp
        command = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {basename} > {clip_name}_duration.txt"
        os.system(command)
        duration_in_second = np.loadtxt(f'{clip_name}_duration.txt', dtype=float)
        duration_in_minutes = duration_in_second / 60.0
        record_json[clip_name] = {}
        record_json[clip_name]['frames'] = duration_in_second * fps
        total_frames_count += record_json[clip_name]['frames']
        record_json[clip_name]['duration'] = f'{duration_in_minutes} minutes'
        # exit(0)
    record_json['total_frames_count'] = total_frames_count
    with open(record_file, 'w') as f:
        json.dump(record_json, f, indent=4)


def create_image_list_file_from_folder():
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508')
    # # batch 0
    # det_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/Raw/obb_images')
    # list_file_batch0 = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/image_list_batch0.txt')
    # list_file_remaining = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/image_list_batch0_remaining.txt')
    # image_list272 = []
    # image_list_remaining = []
    # for file in sorted(glob.glob(os.path.join(det_dir, '*.jpg'))):
    #     basename = os.path.basename(file)
    #     tags = basename.split('_')
    #     channel, image_index = tags[2], tags[-1]
    #     image_index = int(image_index.split('.')[0])
    #     if image_index % 10 == 0:
    #         image_list272.append(basename)
    #     else:
    #         image_list_remaining.append(basename)
    # np.savetxt(list_file_batch0, image_list272, fmt='%s')
    # np.savetxt(list_file_remaining, image_list_remaining, fmt='%s')

    # batch 1
    image_list_remaining = np.loadtxt(os.path.join(dataset_dir, 'image_list_batch0_remaining.txt'), dtype=str)
    list_file_batch1 = os.path.join(dataset_dir, 'image_list_batch1.txt')
    list_file_remaining = os.path.join(dataset_dir, 'image_list_batch1_remaining.txt')
    image_list_batch1 = []
    image_list_batch1_remaining = []
    for file in image_list_remaining:
        image_index = int(os.path.splitext(file)[0].split('_')[-1])
        if image_index % 5 == 0:
            image_list_batch1.append(file)
        else:
            image_list_batch1_remaining.append(file)
    np.savetxt(list_file_batch1, image_list_batch1, fmt='%s')
    np.savetxt(list_file_remaining, image_list_batch1_remaining, fmt='%s')



def rewrite_image_folder_in_prediction_json():
    # https://labelstud.io/guide/storage.html#Local-storage
    # image_folder = ''  # path to image folder exclusing /home/user
    image_folder = "Documents/Work/QUBRF/Piglets/Data/MSUclips/obb_labelling"
    # dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508')
    # prediction_json_file = os.path.join('..', 'image2729_task_with_predictions.json')
    # prediction_json_file_rewritten = os.path.join(dataset_dir, 'image2729_task_with_predictions_local.json')
    # prediction_json_file = os.path.join(dataset_dir, 'project-2-at-2025-07-31-10-15-c5e40779.json')
    # prediction_json_file_rewritten = os.path.join(dataset_dir, 'image_task_test_with_predictions.json')
    # prediction_json_file = os.path.join(dataset_dir, 'project-2-at-2025-07-31-12-53-2600f5b6.json')
    # prediction_json_file_rewritten = os.path.join(dataset_dir, 'image_task_0731_corrections.json')
    json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudiocheck/2508_obb_det')
    prediction_json_file = os.path.join(json_dir, 'project-2-at-2025-08-01-10-27-656c2891.json')
    prediction_json_file_rewritten = os.path.join(json_dir, 'batch1-at-2025-08-01-10-27-656c2891-restructured.json')
    prefix = f"/data/local-files/?d={image_folder}/"
    f = open(prediction_json_file, 'r')
    contents = json.load(f)
    f.close()
    for i, tasks in enumerate(contents):
        image_name = os.path.basename(tasks["data"]["image"])
        contents[i]["data"]["image"] = os.path.join(prefix, image_name)
    with open(prediction_json_file_rewritten, 'w') as f:
        json.dump(contents, f, indent=4)


def filter_prediction_json_with_image_list():
    # dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508')
    # prediction_json_file = os.path.join(dataset_dir, 'image2729_task_with_predictions.json')
    # # image_list_file = os.path.join(dataset_dir, 'image_list_batch0.txt')
    # # filtered_prediction_json_file = os.path.join(dataset_dir, 'image_task_with_predictions_batch0.json')
    # image_list_file = os.path.join(dataset_dir, 'image_list_batch1.txt')
    # filtered_prediction_json_file = os.path.join(dataset_dir, 'image_task_with_predictions_batch1.json')
    # keep_image_list = np.loadtxt(image_list_file, dtype=str)
    # print(f'Number of images to keep: {len(keep_image_list)}')
    # f = open(prediction_json_file, 'r')
    # contents = json.load(f)
    # f.close()
    # task_count = 0
    # filtered_predictions = []
    # for i, tasks in enumerate(contents):
    #     image_name = os.path.basename(tasks["data"]["image"])
    #     if image_name in keep_image_list:
    #         filtered_predictions.append(tasks)
    #         task_count += 1
    #         # print(f'Removed {image_name}')
    # print(f'Number of images kept: {task_count}')
    # with open(filtered_prediction_json_file, 'w') as f:
    #     json.dump(filtered_predictions, f, indent=4)

    # # filter task json with labelled json
    # data_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudiocheck/2508_obb_det')
    # task_json_file = os.path.join(data_dir, 'image_task_with_predictions_batch1.json')
    # labelled_json_file = os.path.join(data_dir, 'project-2-at-2025-08-01-10-27-656c2891.json')
    # updated_task_json_file = os.path.join(data_dir, 'image_task_with_predictions_batch1_exclude56_0805.json')
    batch = 0
    data_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/batch0805')
    task_json_file = os.path.join(data_dir, 'image2553_task_with_predictions.json')
    # labelled_json_file = os.path.join(data_dir, 'consolidated.json')
    # image_list_file = os.path.join(data_dir, 'image_list_batch1.txt')  # ###.jpg
    image_list_file = os.path.join(data_dir, f'image_list_batch{batch}.txt')  # ###.jpg
    batch_list = np.loadtxt(image_list_file, dtype=str)
    # # collect labelled samples
    labelled_images = np.loadtxt(os.path.join(data_dir, 'consolidated_labels_list.txt'), dtype=str)
    labelled_images = [row + '.jpg' for row in labelled_images]
    # labelled_count = 0
    # with open(labelled_json_file, 'r') as f:
    #     labelled_contents = json.load(f)
    #     labelled_images = []
    #     for i, tasks in enumerate(labelled_contents):
    #         image_name = os.path.basename(tasks["data"]["image"])
    #         labelled_images.append(image_name)
    #         labelled_count += 1
    # print(f'Number of tasks labelled: {labelled_count}')

    # scan through all tasks and remove labelled samples
    keep_tasks = []
    drop_count, keep_count = 0, 0
    with open(task_json_file, 'r') as f:
        task_contents = json.load(f)
        for i, tasks in enumerate(task_contents):
            image_name = os.path.basename(tasks["data"]["image"])
            # filter for specific batch
            if image_name not in batch_list:
                continue
            if image_name in labelled_images:
                drop_count += 1
            else:
                keep_tasks.append(tasks)
                keep_count += 1
    print(f'Number of tasks dropped: {drop_count}')
    print(f'Number of tasks kept: {keep_count}')
    updated_task_json_file = os.path.join(data_dir, f'image_task_with_predictions_batch{batch}_0805remain{keep_count}.json')
    with open(updated_task_json_file, 'w') as f:
        json.dump(keep_tasks, f, indent=4)


def check_out_of_image_interpolation():
    num_frame = 15
    min_key_frame = 1
    clip_name = 'ch12_20250109090000_003000'
    check_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudio_export_check')
    # step 1: separate annotations for clip from project
    in_json = os.path.join(check_dir, 'project-19-at-2025-07-31-11-57-b1dae767.json')
    filtered_prediction_json_file = os.path.join(check_dir, f'{clip_name}.json')
    f = open(in_json, 'r')
    contents = json.load(f)
    f.close()
    # filtered_predictions = []
    for i, tasks in enumerate(contents):
        image_name = os.path.basename(tasks["data"]["video"])
        if image_name == clip_name + '.mp4':
            # anno_result = tasks["annotations"]["result"]
            # for anno in anno_result:
            #     sequence = anno['sequence']
            #     for seq in sequence:
            #         if int(seq['frame']) < num_frame:
            # filtered_predictions.append(tasks)
            with open(filtered_prediction_json_file, 'w') as f:
                json.dump(tasks, f, indent=4)
            break
    # step 2: calculate interpolations
    input_json_path = os.path.join(check_dir, f'{clip_name}.json')
    output_json_path = os.path.join(check_dir, f'{clip_name}_interpolation_check.json')
    ls_io_utils.process_labelstudio_interpolation(input_json_path, output_json_path, min_key_frame)


def create_piglet_tracklab_dataset():
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
    img_w, img_h = 2688, 1520
    fps = 30
    data_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507')
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/tracklab')
    for split, clips in clips.items():
        split_dir = os.path.join(dataset_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        image_source_dir = os.path.join(data_dir, f'obb_detection/images/{split}')
        for clip_name in clips:
            seq_folder = os.path.join(split_dir, clip_name)
            # prepare images in img1 folder ---------------------------------
            image_folder = os.path.join(seq_folder, 'img1')
            os.makedirs(seq_folder, exist_ok=True)
            os.makedirs(image_folder, exist_ok=True)
            # copy images into individual seq/sample folder
            os.chdir(image_source_dir)
            frames_count = 0
            for file in sorted(glob.glob(os.path.join(image_source_dir, f'{clip_name}_*.jpg'))):
                file_name = os.path.basename(file)
                image_file = os.path.join(image_folder, file_name)
                if not os.path.isfile(image_file):
                    command = f'cp {file_name} ../../../tracklab/{split}/{clip_name}/img1/'
                    os.system(command)
                    print(f'Copied  {file_name} into dataset {split} {clip_name}')
                frames_count += 1
            # print(f'Finished collecting {frames_count} images into dataset {split} split')

            # # prepare seqinfo.ini file ---------------------------------
            # seginfo_file = os.path.join(seq_folder, 'seqinfo.ini')
            # with open(seginfo_file, 'w') as writer:
            #     writer.write('[Sequence]\n')
            #     writer.write(f'name={clip_name}\n')
            #     writer.write(f'imDir=img1\n')
            #     writer.write(f'frameRate={fps}\n')
            #     writer.write(f'seqLength={frames_count}\n')
            #     writer.write(f'imWidth={img_w}\n')
            #     writer.write(f'imHeight={img_h}\n')
            #     writer.write(f'imExt=.jpg\n')
            #     writer.flush()
            # print(f'Finished writing {seginfo_file}')

            # # prepare gt.txt file ---------------------------------
            # gt_folder = os.path.join(seq_folder, 'gt')
            # os.makedirs(gt_folder, exist_ok=True)
            # gt_file_to_write = os.path.join(gt_folder, 'gt.txt')
            # in_gt_file = os.path.join(data_dir, f'{clip_name}_mot.txt')
            # gt_mot = np.loadtxt(in_gt_file, delimiter=',', dtype=int)
            # # expected gt file format
            # # ['image_id', 'track_id', 'left', 'top', 'width', 'height', 'bbox_conf', 'class', 'visibility']
            # gt_tracklab = gt_mot[:, :6]
            # gt_tracklab = np.hstack((gt_tracklab, np.ones((gt_tracklab.shape[0], 3), dtype=np.int32)))
            # # np.savetxt(gt_file_to_write, gt_tracklab)
            # with open(gt_file_to_write, 'w') as fw:
            #     for gt in gt_tracklab:
            #         fw.write(f'{gt[0]:0d},1,{gt[2]:0d},{gt[3]:0d},{gt[4]:0d},{gt[5]:0d},1,1,1\n')
            #         fw.flush()
            # print(f'Finished writing {gt_file_to_write}')

            # # prepare det.txt file ---------------------------------
            # det_folder = os.path.join(seq_folder, 'det')
            # os.makedirs(det_folder, exist_ok=True)
            # det_file = os.path.join(det_folder, 'det.txt')
            # model_dir = os.path.join(ROOT_DIR, 'WTData/Results/250722_yolo11s_obb/train')
            # result_csv_file = os.path.join(model_dir, f'yolo11n-obb-{clip_name}-predictions-filtered.txt')
            # data_utils.convert_yolo_obb_result_to_mot_det_save_file(result_csv_file, det_file)
            # print(f'Finished writing {det_file}')


def copy_piglet2507_data_as_piglet2508_val_split():
    # note: we cannot use tracking frames only for validation, because side pens are not labelled
    splits_to_copy = ['train', 'val', ]
    source_dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection')
    new_dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/batch0805')
    new_image_dir = os.path.join(new_dataset_dir, 'images/val')
    new_label_dir = os.path.join(new_dataset_dir, 'labels/val')
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)
    val_count = 0
    for split in splits_to_copy:
        source_image_dir = os.path.join(source_dataset_dir, 'images', split)
        source_label_dir = os.path.join(source_dataset_dir, 'labels', split)
        for file in sorted(glob.glob(os.path.join(source_image_dir, '*.jpg'))):
            file_name = os.path.basename(file)
            file_name = os.path.splitext(file_name)[0]
            frame = int(file_name.split('_')[-1])
            if frame % 60 == 1:
                os.chdir(source_image_dir)
                command = f"cp {file_name}.jpg ../../../../Piglet2508/batch0805/images/val/"
                os.system(command)
                os.chdir(source_label_dir)
                command = f"cp {file_name}.txt ../../../../Piglet2508/batch0805/labels/val/"
                os.system(command)
                val_count += 1
                print(f'Copying {file_name} ')
    print(f'Finished copying {val_count} samples to {new_dataset_dir} val split')


if __name__ == '__main__':
    # check_image_obb_json()
    # extract_frames_for_obb_labelling()
    # check_video_length()
    # remove_tracking_frames_from_obb_labelling_images()
    # create_image_list_file_from_folder()
    filter_prediction_json_with_image_list()
    # rewrite_image_folder_in_prediction_json()
    # check_out_of_image_interpolation()
    # create_piglet_tracklab_dataset()
    # check_image_obb_json()
    # copy_piglet2507_data_as_piglet2508_val_split()