import os
import cv2
import glob
import math
import json
import numpy as np
import ls_io_utils
from deepdiff import DeepDiff
from config import ROOT_DIR, BOX_COLORS
import visualisations


def extract_first_frame_for_check():
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'Raw')
    completed_channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6']
    os.chdir(data_dir)
    for file in glob.glob(os.path.join(data_dir, '*.mp4')):
        basename = os.path.basename(file)
        tags = basename.split('_')
        channel = tags[2]
        if channel in completed_channels:
            continue
        timestamp = tags[4]
        image_name = channel + '_' + timestamp + '_00001.png'
        command = f'ffmpeg -i {basename} -vf "select=eq(n\,0)" -vsync 0 -frames:v 1 {image_name}'
        os.system(command)
        print(f'Image extracted: {image_name}')


def extract_clip_at_given_length():
    dara_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'Raw')
    extract_channels = ['ch3', 'ch6', 'ch8', 'ch9', 'ch11', 'ch12', 'ch14', 'ch16', 'ch17']
    start_minute = [0, 5, 15, 20, 25, 30, 35, 40, 45]
    os.chdir(dara_dir)
    for file in glob.glob(os.path.join(dara_dir, '*.mp4')):
        basename = os.path.basename(file)
        tags = basename.split('_')
        channel = tags[2]
        if channel not in extract_channels:
            continue
        timestamp = tags[4]
        for sm in start_minute:
            clip_name = channel + '_' + timestamp + f'_00{sm:02d}00.mp4'
            if os.path.exists(clip_name):
                continue
            command = f'ffmpeg -i {basename} -ss 00:{sm:02d}:00 -t 00:01:00 -c:v libx264 -r 30 -an -strict experimental -movflags +faststart {clip_name}'
            os.system(command)
            print(f'Clip extracted: {clip_name}')


def extract_frames_from_clip():
    clips = {
        # 'train': ['ch8_20250109093748_000000', 'ch11_20250109094853_000500', 'ch12_20250109090000_000200',
        #           'ch12_20250109090000_003000'],
        'train': ['ch2_20250109090000_000200', ]#'ch6_20250109093801_000500']  # new added training sequences 0716
        # 'val': ['ch1_20250109090000_001900', ]#'ch12_20250109090000_001700']
    }
    clip_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/clips')
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507')
    os.makedirs(dataset_dir, exist_ok=True)
    image_save_dir = os.path.join(dataset_dir, 'obb_detection/images')
    os.makedirs(image_save_dir, exist_ok=True)
    for split, clips in clips.items():
        image_split_dir = os.path.join(dataset_dir, f'obb_detection/images/{split}')
        os.makedirs(image_split_dir, exist_ok=True)
        for clip_name in clips:
            # first, check and remove existing extracted frames in the folder
            os.chdir(image_split_dir)
            frame_count = 0
            for file in sorted(glob.glob(os.path.join(image_split_dir, f'{clip_name}_*.jpg'))):
                basename = os.path.basename(file)
                command = f'rm {basename}'
                os.system(command)
                frame_count += 1
            print(f'{clip_name}: removed {frame_count} frames')
            # secondly, copy video clip into dataset folder
            clip_path = os.path.join(dataset_dir, f'{clip_name}.mp4')
            if not os.path.exists(clip_path):
                command = f'cp {clip_name}.mp4 ../../Dataset/Piglet2507'
                os.chdir(clip_dir)
                os.system(command)
            # thirdly, extract frames from video clip into dataset split folder, forcing fps30
            command = f'ffmpeg -i {clip_name}.mp4 -vf fps=30 obb_detection/images/{split}/{clip_name}_%04d.jpg'
            os.chdir(dataset_dir)
            os.system(command)
            # fourthly, check number of frames extracted
            frame_count = 0
            for file in sorted(glob.glob(os.path.join(image_split_dir, f'{clip_name}_*.jpg'))):
                frame_count += 1
            print(f'Video extracted: {clip_name} in {split} split with {frame_count} frames')


def check_individual_piglet_frame():
    objs_of_interest = ['p9']
    frames_of_interest = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ]
    dara_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'labelstudiocheck')
    ls_json = os.path.join(dara_dir, 'ch2_20250109090000_000200_incorrect_rotation.json')
    with open(ls_json, 'r') as f:
        contents = json.load(f)
        anno_result = contents["annotations"][0]["result"]
        for objects in anno_result:  # loop through all labelled objects
            anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
            label = anno_value["labels"][0]  # str
            # obj_id = int(label.replace('p', ''))
            if label not in objs_of_interest:
                continue
            for entry in anno_value["sequence"]:  # loop through all labelled key frames
                frame_id = int(entry['frame'])
                if frame_id not in frames_of_interest:
                    continue
                r = int(entry['rotation'])
                x = entry['x']
                y = entry['y']
                w = entry['width']
                h = entry['height']
                print(f'{label} frame{frame_id}: x:{x:.6f}, y:{y:.6f}, w:{w:.6f}, h:{h:.6f}, r:{r:.6f}')


def ls_json_to_yolo_obb():
    spec_frame = -1
    clips = {
        # 'train': ['ch8_20250109093748_000000', 'ch11_20250109094853_000500', 'ch12_20250109090000_000200',
        #           'ch12_20250109090000_003000'],
        # 'train': ['ch2_20250109090000_000200', 'ch6_20250109093801_000500']  # new added training sequences 0716
        # 'val': ['ch1_20250109090000_001900', 'ch12_20250109090000_001700']
        'val': ['ch1_20250109090000_001900', ]  # separate processing because of missing images
    }
    img_w, img_h = 2688.0, 1520.0
    json_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'video_obb_annotations')
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507')
    for split, clips in clips.items():
        image_dir = os.path.join(dataset_dir, f'obb_detection/images/{split}')
        obb_label_dir = os.path.join(dataset_dir, f'obb_detection/labels/{split}')
        os.makedirs(obb_label_dir, exist_ok=True)
        for clip_name in clips:
            # fps25 = False if clip_name == 'ch12_20250109090000_000200' else True
            # if fps25:
            #     anno_json_file = os.path.join(data_dir, f'{clip_name}_fps25.json')
            # else:
            # anno_json_file = os.path.join(data_dir, f'{clip_name}.json')
            anno_json_file = os.path.join(json_dir, f'{clip_name}_with_interpolations.json')
            if not os.path.exists(anno_json_file):
                print(f'Skipping because anno JSON not found: {anno_json_file}')
                continue
            label_file_count = ls_io_utils.write_ls_video_export_json_into_yolo_obb(anno_json_file, obb_label_dir,
                                                                                    image_dir, img_w, img_h, spec_frame)
            print(f'Label files written for {clip_name}: {label_file_count}')


def draw_verify_yolo_obb():
    frame_id = -1#000
    img_w, img_h = 2688.0, 1520.0
    clips = {
        'train': ['ch8_20250109093748_000000', 'ch11_20250109094853_000500', 'ch12_20250109090000_000200',
                  'ch12_20250109090000_003000', 'ch2_20250109090000_000200', 'ch6_20250109093801_000500'],
        # 'train': ['ch2_20250109090000_000200', ]#'ch6_20250109093801_000500']  # new added training sequences 0716
        # 'val': ['ch12_20250109090000_001700', ]#'ch1_20250109090000_001900', 'ch12_20250109090000_001700']
    }
    # vis_dir = os.path.join(ROOT_DIR, f'WTData/Dataset/Piglet2506/obb_detection')
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection')
    vis_dir = os.path.join(dataset_dir, 'labelcheck')
    os.makedirs(vis_dir, exist_ok=True)
    for split, clips in clips.items():
        image_dir = os.path.join(dataset_dir, f'images/{split}')
        label_dir = os.path.join(dataset_dir, f'labels/{split}')
        for clip_name in clips:
            if frame_id < 0:
                for i in range(1, 1801):
                    vis_image = visualisations.draw_yolo_obb_label_on_image(image_dir, label_dir, clip_name, i, img_w, img_h)
                    if vis_image is not None:
                        vis_file = os.path.join(vis_dir, f'labelcheck_{split}_{clip_name}_{i:04d}.jpg')
                        cv2.imwrite(vis_file, vis_image)
                command = f'ffmpeg -framerate 30 -i labelcheck_{split}_{clip_name}_%04d.jpg -c:v libx264 -pix_fmt yuv420p ../labelcheck_{split}_{clip_name}.mp4'
                os.chdir(vis_dir)
                os.system(command)
            else:
                vis_image = visualisations.draw_yolo_obb_label_on_image(image_dir, label_dir, clip_name, frame_id, img_w, img_h)
                vis_file = os.path.join(vis_dir, f'labelcheck_{split}_{clip_name}_{frame_id:04d}.jpg')
                cv2.imwrite(vis_file, vis_image)


def re_write_ls_json_fps25_to30():
    clips = [# 'ch12_20250109090000_003000',
             #'ch12_20250109090000_001700',
             # 'ch8_20250109093748_000000',
             # 'ch1_20250109090000_001900',
             'ch2_20250109090000_000200',
            ]
    # in_json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudiocheck')
    # out_json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudiocheck')#video_obb_annotations')
    in_json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudio_export_check')
    # out_json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudio_export_check')
    out_json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/video_obb_annotations')
    # in_json_file = os.path.join(in_json_dir, 'ch12_20250109090000_001700_fps25.json')
    # out_json_file = os.path.join(out_json_dir, 'ch12_20250109090000_001700_fps30.json')
    # in_json_file = os.path.join(in_json_dir, 'ch8_20250109093748_000000_fps25.json')
    # out_json_file = os.path.join(out_json_dir, 'ch8_20250109093748_000000_fps30.json')
    for clip_name in clips:
        in_json_file = os.path.join(in_json_dir, f'{clip_name}_fps25_no_interpolations.json')
        # out_json_file = os.path.join(out_json_dir, f'{clip_name}_no_interpolations.json')
        out_json_file = os.path.join(out_json_dir, f'{clip_name}_keyframes.json')
        with open(in_json_file, 'r') as f:
            contents = json.load(f)
            anno_result = contents["annotations"][0]["result"]
            for i, objects in enumerate(anno_result):
                anno_value = objects["value"]
                total_frames = anno_value["framesCount"]
                anno_value["framesCount"] = math.ceil(total_frames / 25. * 30.)  # rounding up
                for j, entry in enumerate(anno_value["sequence"]):
                    frame_id = int(entry['frame'])
                    new_frame_id = int(frame_id / 25. * 30.)
                    contents["annotations"][0]["result"][i]["value"]["sequence"][j]['frame'] = new_frame_id

            with open(out_json_file, 'w') as f:
                json.dump(contents, f, indent=4)


def remove_incorrect_yolo_labels():
    clip_to_remove = 'ch12_20250109090000_003000'  #'ch8_20250109093748_000000'  #
    split = 'train'
    label_dir = os.path.join(ROOT_DIR, f'WTData/Dataset/Piglet2506/obb_detection/labels/{split}')
    image_dir = os.path.join(ROOT_DIR, f'WTData/Dataset/Piglet2506/obb_detection/images/{split}')
    # os.chdir(label_dir)
    # for file in sorted(glob.glob(os.path.join(label_dir, f'{clip_to_remove}_*.txt'))):
    #     basename = os.path.basename(file)
    #     command = f'rm {basename}'
    #     os.system(command)
    #     print(f'Label file removed: {basename}')
    os.chdir(image_dir)
    for file in sorted(glob.glob(os.path.join(image_dir, f'{clip_to_remove}_*.jpg'))):
        basename = os.path.basename(file)
        command = f'rm {basename}'
        os.system(command)
        print(f'Image file removed: {basename}')


def compare_ls_export_json():
    """Compare two JSON files and print differences."""
    data_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudio_export_check/')
    # json_file1 = os.path.join(data_dir, 'project-21-at-2025-07-15-13-10-f08ded08.json')  # incorrect rotation
    # json_file2 = os.path.join(data_dir, 'project-21-at-2025-07-15-14-14-f08ded08.json')
    json_file1 = os.path.join(data_dir, 'ch12_20250109090000_003000_wrong_interpolations.json')
    json_file2 = os.path.join(data_dir, 'ch12_20250109090000_003000_with_interpolations.json')
    f1 = open(json_file1, 'r', encoding='utf-8')
    json1 = json.load(f1)
    f1.close()
    f2 = open(json_file2, 'r', encoding='utf-8')
    json2 = json.load(f2)
    f2.close()
    diff = DeepDiff(json1, json2, ignore_order=True)

    if not diff:
        print("✅ The JSON files are identical.")
    else:
        print("⚠️ Differences found:")
        print(json.dumps(diff, indent=4))


def get_interpolated_obb_from_json():
    min_key_frame = 1
    clip_names = [# 'ch1_20250109090000_001900',
                  'ch2_20250109090000_000200',
                  # 'ch6_20250109093801_000500',
                  # 'ch8_20250109093748_000000',
                  # 'ch11_20250109094853_000500',
                  # 'ch12_20250109090000_000200',
                  # 'ch12_20250109090000_001700',
                  # 'ch12_20250109090000_003000',
                  ]
    # data_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudio_export_check/')
    data_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/video_obb_annotations/')
    save_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/full_annotation')
    os.makedirs(save_dir, exist_ok=True)
    for clip in clip_names:
        # input_json_path = os.path.join(data_dir, f'{clip}_no_interpolations.json')
        input_json_path = os.path.join(data_dir, f'{clip}_keyframes.json')
        output_json_path = os.path.join(save_dir,  f'{clip}_with_interpolations.json')
        ls_io_utils.process_labelstudio_interpolation(input_json_path, output_json_path, min_key_frame)


def draw_verify_ls_obb_json():
    spec_frame = 1800  # -1  #  888 # 1799 # # 1799  # 777
    img_w, img_h = 2688.0, 1520.0
    clips = {
        'train': ['ch2_20250109090000_000200',  # extra 1 minute
                  # 'ch6_20250109093801_000500',  # missing boxes on frame 1800
                  # 'ch8_20250109093748_000000',  # missing boxes on frame 1800
                  # 'ch11_20250109094853_000500',  # vertically flipped, missing boxes on frame 1800
                  # 'ch12_20250109090000_000200',  # missing boxes on frame 1800, static piglet removed, frame rotated
                  # 'ch12_20250109090000_003000',  # checked
                  ],
        # 'val': ['ch1_20250109090000_001900',  # missing boxes on frame 1799
        #         'ch12_20250109090000_001700',  # checked
        #         ]
    }
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection')
    # json_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'video_obb_annotations')
    json_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'full_annotation')
    vis_dir = os.path.join(dataset_dir, 'jsonlabelcheck0718')
    os.makedirs(vis_dir, exist_ok=True)
    for split, clips in clips.items():
        image_dir = os.path.join(dataset_dir, f'images/{split}')
        for clip_name in clips:
            cam = clip_name.split('_')[0]
            anno_json_file = os.path.join(json_dir, f'{clip_name}_with_interpolations.json')
            json_reader = open(anno_json_file, 'r')
            contents = json.load(json_reader)
            json_reader.close()
            anno_result = contents["annotations"][0]["result"]
            check_video = f'jsoncheck_{split}_{clip_name}.mp4'
            # if spec_frame < 0 and os.path.exists(os.path.join(dataset_dir, check_video)):
            #     print(f'Skipping {check_video} ...')
            #     continue
            for file in sorted(glob.glob(os.path.join(image_dir, f'{clip_name}_*.jpg'))):
                file_name = os.path.basename(file)
                file_name = os.path.splitext(file_name)[0]
                frame_id = int(file_name.split('_')[-1])
                if 0 < spec_frame != frame_id:
                    continue
                vis_file = os.path.join(vis_dir, f'jsoncheck_{split}_{clip_name}_{frame_id:04d}.jpg')
                image = cv2.imread(file)
                # loop through all labelled objects because LS JSON is organised by objects
                for objects in anno_result:
                    anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
                    obj_id = anno_value["labels"][0]  # str, i.e., p1
                    for entry in anno_value["sequence"]:  # loop through all labelled key frames
                        frame_json = int(entry['frame'])
                        # find object json entry for this frame
                        if frame_json != frame_id:
                            continue
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
                        x1, x2, x3, x4 = int(xtl), int(P1[0]), int(P2[0]), int(P3[0])
                        y1, y2, y3, y4 = int(ytl), int(P1[1]), int(P2[1]), int(P3[1])
                        color = BOX_COLORS[obj_id]  # obj_id: p1
                        cv2.line(image, (x1, y1), (x2, y2), color, thickness=2)
                        cv2.line(image, (x2, y2), (x3, y3), color, thickness=2)
                        cv2.line(image, (x3, y3), (x4, y4), color, thickness=2)
                        cv2.line(image, (x1, y1), (x4, y4), color, thickness=2)
                        # xc = int((x1 + x2 + x3 + x4) / 4)
                        # yc = int((y1 + y2 + y3 + y4) / 4)
                        cv2.putText(image, obj_id, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    color=(255, 255, 255), thickness=3)
                        cv2.putText(image, obj_id, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    color=color, thickness=2)
                        # cv2.putText(image, obj_id, (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #             color=(255, 255, 255), thickness=3)
                        # cv2.putText(image, obj_id, (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #             color=color, thickness=2)
                        # end loop within object when this frame found
                        if frame_json == frame_id:
                            break
                cv2.putText(image, f'{cam} frame {frame_id}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            color=(255, 255, 255), thickness=3)
                cv2.imwrite(vis_file, image)
                if frame_id % 300 == 0:
                    print(f'Image written to {vis_file}')
                # end loop through frames when specific frame found
                if 0 < spec_frame == frame_id:
                    print(f'Exiting with frame {frame_id} found')
                    break
            # if spec_frame < 0:
            #     command = f'ffmpeg -framerate 30 -i jsoncheck_{split}_{clip_name}_%04d.jpg -c:v libx264 -pix_fmt yuv420p ../{check_video}'
            #     os.chdir(vis_dir)
            #     os.system(command)


def concatenate_labelcheck_images_to_mp4():
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection')
    vis_dir = os.path.join(dataset_dir, 'jsonlabelcheck0718')
    clips = {
        'train': ['ch2_20250109090000_000200',  # extra 1 minute
                  # 'ch6_20250109093801_000500',  # json checked
                  # 'ch8_20250109093748_000000',  # json checked
                  # 'ch11_20250109094853_000500',  # json checked
                  # 'ch12_20250109090000_000200',  # json checked
                  # 'ch12_20250109090000_003000',  # json checked
                  ],
        # 'val': ['ch1_20250109090000_001900',  # json checked
        #         'ch12_20250109090000_001700',  # json checked
        #         ]
    }
    for split, clips in clips.items():
        image_dir = os.path.join(dataset_dir, f'images/{split}')
        for clip_name in clips:
            check_video = f'jsoncheck_{split}_{clip_name}.mp4'
            # frame_count = 0
            # for file in sorted(glob.glob(os.path.join(image_dir, f'{clip_name}_*.jpg'))):
            #     frame_count += 1
            # print(f'{clip_name}: {frame_count} frames in {image_dir}')
            # frame_count = 0

            # os.chdir(vis_dir)
            # for file in sorted(glob.glob(os.path.join(vis_dir, f'jsoncheck_{split}_{clip_name}_*.jpg'))):
            #     basename = os.path.basename(file)
            #     command = f'rm {basename}'
            #     os.system(command)
            #     frame_count += 1
            # print(f'{clip_name}: {frame_count} frames in {vis_dir} removed')

            command = f'ffmpeg -framerate 30 -i jsoncheck_{split}_{clip_name}_%04d.jpg -c:v libx264 -pix_fmt yuv420p ../{check_video}'
            os.chdir(vis_dir)
            os.system(command)


def separate_json():
    video_interest = ''
    bulk_json_file = ''
    bulk_json_file = os.path.join('', bulk_json_file)
    ls_io_utils.separate_annotation_per_video(bulk_json_file, video_interest)




def calculate_pigle_dataset_utils():
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
    stats = {}
    dataset_dir = os.path.join(ROOT_DIR, 'WTData', 'Dataset', 'Piglet2507')
    for split, clips in clips.items():
        stats[split] = {}
        stats[split]['num_tracks'] = 0
        stats[split]['num_boxes'] = 0
        stats[split]['num_frames'] = 0
        for clip_name in clips:
            sample_csv_file = os.path.join(dataset_dir, f'{clip_name}_mot.txt')
            all_tracks = np.loadtxt(sample_csv_file, delimiter=',', dtype=np.int32)
            num_boxes = len(all_tracks)
            num_tracks = len(np.unique(all_tracks[:, 1]))
            num_frames = len(np.unique(all_tracks[:, 0]))
            stats[split]['num_boxes'] += num_boxes
            stats[split]['num_tracks'] += num_tracks
            stats[split]['num_frames'] += num_frames
    print(stats)
    with open(os.path.join(dataset_dir, 'pigle_dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)




def run_piglet_detector_save_result_into_ls_json():
    pass




def create_ls_json_image_obb_with_choice_template():
    in_json_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/labelstudiocheck')
    in_json_file = os.path.join(in_json_dir, 'project-24-restructured.json')
    anno_file_to_write = os.path.join(in_json_dir, '250725_image_obb_choice_import_test.json')
    prelabels = {}
    prelabels["data"] = {}
    prelabels["predictions"] = []
    with open(in_json_file, 'r') as f:
        contents = json.load(f)  # all tasks / images
        for cont in contents:
            for k, v in cont["data"].items():
                prelabels["data"][k] = v
            annotations = cont["annotations"]
            for anno in annotations:
                result = anno["result"]  # all the labelled objects
                entry = {}
                entry["result"] = result
                entry["model_version"] = 'yolo11n-obb'
                entry["score"] = 0.6
                prelabels["predictions"].append(entry)

    with open(anno_file_to_write, 'w') as fw:
        json.dump(prelabels, fw, indent=4)


def convert_yolo_obb_result_to_mot_det_save_file(yolo_obb_file, mot_det_file):
    # OBB result format: [frame_id, conf, x1, y1, x2, y2, x3, y3, x4, y4]
    yolo_obbs = np.loadtxt(yolo_obb_file, delimiter=',')
    # expected det file format
    # ['image_id', 'track_id'(-1), 'left', 'top', 'width', 'height', 'bbox_conf', 'x'(-1), 'y'(-1), 'z'(-1)]
    image_ids = yolo_obbs[:, 0].astype(np.int32)# np.expand_dims(yolo_obbs[:, 0].astype(np.int32), axis=1)
    bbox_confs = yolo_obbs[:, 1].astype(np.float32)# np.expand_dims(yolo_obbs[:, 1].astype(np.float32), axis=1)
    xs = yolo_obbs[:, [2,4,6,8]]
    ys = yolo_obbs[:, [3,5,7,9]]
    lefts = np.min(xs, axis=1)#[:, np.newaxis]
    tops = np.min(ys, axis=1)#[:, np.newaxis]
    widths = np.max(xs, axis=1) - lefts
    heights = np.max(ys, axis=1) - tops
    # negatives = np.full((len(bbox_confs), 1), -1, dtype=np.int32)
    # dets = np.hstack((image_ids, negatives, lefts, tops, widths, heights, bbox_confs, negatives, negatives, negatives))
    with open(mot_det_file, 'w') as fw:
        for i in range(len(bbox_confs)):
            fw.write(f'{image_ids[i]:0d},-1,{lefts[i]:.2f},{tops[i]:.2f},{widths[i]:.2f},{heights[i]:.2f},{bbox_confs[i]:.6f},-1,-1,-1\n')
            fw.flush()
    # np.savetxt(mot_det_file, dets)


def create_image_obb_predictions():
    # step 1: extract images from videos - done
    # step 2: collect all images into LS task import json - done
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508')
    image_folder = "Documents/Work/QUBRF/Piglets/Data/MSUclips/obb_labelling"
    # json_file_to_write = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/image_task.json')
    json_file_to_write = os.path.join(dataset_dir, 'image2729_task_empty_predictions.json')
    # image_list = [f"ch6_20250109093801_{i:04d}.jpg" for i in range(1, 5)]
    image_list = [
        'ch3_20250109090000_0001.jpg',
        'ch4_20250109093747_0004.jpg',
        'ch7_20250109090000_0006.jpg',
        'ch12_20250109090000_0108.jpg',
        'ch16_20250109090002_0064.jpg',
    ]
    full_image_list = np.loadtxt(os.path.join(dataset_dir, 'image_list.txt'), dtype=str)
    # ls_io_utils.write_image_task_json_from_list(image_list, image_folder, json_file_to_write)
    ls_io_utils.write_image_prediction_json_template_from_image_list(full_image_list, image_folder, json_file_to_write)
    # step 3: run yolo and append OBBs to LS exported empty json - done
    # step 4: manually correct OBBs in LS, export corrected OBBs in json - ongoing
    # step 5: convert json to yolo OBB format, consolidate into detection dataset


def load_multiple_ls_json_consolidate_into_one():
    json_files_to_consplidate = [
        'project-2-at-2025-07-31-10-15-c5e40779',
        'project-2-at-2025-07-31-12-53-2600f5b6',
        'project-2-at-2025-08-01-10-27-656c2891',
        'project-2-at-2025-08-04-11-51-e51535cc',
    ]
    json_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/batch0805')
    out_json = os.path.join(json_dir, 'consolidated.json')
    labelled_images = {}
    labelled_count = 0
    labelled_tasks = {}
    image_folder = "Documents/Work/QUBRF/Piglets/Data/MSUclips/obb_labelling"
    prefix = f"/data/local-files/?d={image_folder}/"
    for file in json_files_to_consplidate:
        json_file = os.path.join(json_dir, file + '.json')
        with open(json_file, 'r') as f:
            labelled_contents = json.load(f)
            for i, tasks in enumerate(labelled_contents):
                image_name = os.path.basename(tasks["data"]["image"])
                timestamp = tasks["updated_at"].split('.')[0]
                if image_name not in labelled_tasks.keys():
                    labelled_tasks[image_name] = tasks
                    labelled_tasks[image_name]["data"]["image"] = os.path.join(prefix, image_name)
                    labelled_images[image_name] = timestamp
                    labelled_count += 1
                else:
                    first_time = labelled_images[image_name]
                    later_ind = ls_io_utils.compare_ls_timestamp_and_find_later(first_time, timestamp)
                    if later_ind == 2:
                        labelled_tasks[image_name] = tasks
                        labelled_tasks[image_name]["data"]["image"] = os.path.join(prefix, image_name)
                        print(f'Found duplicates for {image_name}: {first_time} (discard) and {timestamp}')
    labelled_tasks = list(labelled_tasks.values())
    with open(out_json, 'w') as fw:
        json.dump(labelled_tasks, fw, indent=4)
    print(f'Finish saving json at {out_json} with {labelled_count} images')


def create_obb_det_dataset_from_ls_annotations():
    image_source_dir = os.path.join(ROOT_DIR, 'WTData/Piglet/Raw/obb_images')
    dataset_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2508/batch0805')
    annotated_json_file = os.path.join(dataset_dir, 'consolidated.json')
    label_save_dir = os.path.join(dataset_dir, 'labels/train')
    image_save_dir = os.path.join(dataset_dir, 'images/train')
    os.makedirs(label_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    img_w, img_h = 2688, 1520
    # yaml_dir = os.path.join(ROOT_DIR, 'WTData/Dataset/Piglet2507/obb_detection')
    # command = "cp data.yaml ../../Piglet2508/batch0805"
    # os.chdir(yaml_dir)
    # os.system(command)
    num_labels = ls_io_utils.convert_ls_image_export_json_into_yolo_obb_and_copy_images(annotated_json_file,
                                                                                        label_save_dir, image_save_dir,
                                                                                        image_source_dir, img_w, img_h)
    print("Number of labels: {}".format(num_labels))


if __name__ == '__main__':
    # extract_first_frame_for_check()
    # extract_clip_at_given_length()
    # check_individual_piglet_frame()
    # re_write_ls_json_fps25_to30()
    # remove_incorrect_yolo_labels()
    # compare_ls_export_json()
    # extract_frames_from_clip()
    # ls_json_to_yolo_obb()
    # draw_verify_yolo_obb()
    # get_interpolated_obb_from_json()
    # draw_verify_ls_obb_json()
    # concatenate_labelcheck_images_to_mp4()
    # ls_json_to_yolo_obb0722()
    # ls_json_to_mot_csv250722()
    # calculate_pigle_dataset_utils()
    # find_detection_post_processing_pen_areas()
    # check_soworientation_dataset()
    # create_ls_json_image_obb_with_choice_template()
    # create_image_obb_predictions()
    # load_multiple_ls_json_consolidate_into_one()
    create_obb_det_dataset_from_ls_annotations()