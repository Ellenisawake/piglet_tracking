import os
import cv2
import json
import math
import argparse
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from config import ROOT_DIR, BOX_COLORS
import ls_io_utils
import image_utils




def convert_mask_to_image():
    result_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250414_piglet_sam')
    mask_file = os.path.join(result_dir, 'masks.npy')
    result_img_file = os.path.join(result_dir, 'pigsam.png')
    # mask is a binary array indicating object or background
    masks = np.load(mask_file)
    h, w = masks.shape[-2:]
    result_img = np.zeros((h, w, 3), dtype=np.uint8)
    # color = np.array([30, 144, 255])
    for m in masks:
        color = np.concatenate([np.random.random(3)*255], axis=0)
        mask_image = m.reshape(h, w, 1) * color.reshape(1, 1, -1)
        result_img += mask_image.astype(np.uint8)
    cv2.imwrite(result_img_file, result_img)
    print('Finish converting')


def overlay_image_mask_for_visual():
    image_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'ch3_250109090000_005200_0001.png')
    mask_file = os.path.join(ROOT_DIR, 'WTData', 'Results', '250414_piglet_sam', 'pigsam.png')
    result_img_file = os.path.join(ROOT_DIR, 'WTData', 'Results', '250414_piglet_sam', 'pigsam_overlay.png')
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file)
    result_img = cv2.addWeighted(image, 0.6, mask, 0.4, 0.0)
    cv2.imwrite(result_img_file, result_img)
    print('Finish converting')


def draw_box_mask_for_visual():
    task_id = -1
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet')
    result_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250414_piglet_sam')
    image_file = os.path.join(data_dir, 'ch11_20250109090000_003120_0001.png')
    anno_file = os.path.join(data_dir, 'ch11_anno.json')
    mask_file = os.path.join(result_dir, 'ch11_0001_masks.npy')
    result_img_file = os.path.join(result_dir, 'ch11_0001_pigsam_everything_overlay.png')
    image = cv2.imread(image_file)
    img_h, img_w = image.shape[:2]
    result_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    input_boxes = ls_io_utils.get_boxes_from_anno_json(anno_file, img_h, img_w, task_id)
    predicted_masks = np.load(mask_file)
    for i, (ibox, mask) in enumerate(zip(input_boxes, predicted_masks)):
        obox = image_utils.get_min_area_rect_from_mask_array(mask, img_h, img_w)
        color = np.concatenate([np.random.random(3)*255], axis=0)
        mask_image = mask.reshape(img_h, img_w, 1) * color.reshape(1, 1, -1)
        result_img += mask_image.astype(np.uint8)
        x0, y0, x1, y1 = ibox
        cv2.rectangle(result_img, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
        cv2.drawContours(result_img, [obox], 0, color=color, thickness=2)
        cv2.putText(result_img, str(i), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 255, 255), thickness=3)
        cv2.putText(result_img, str(i), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2)
    result_img = cv2.addWeighted(image, 0.6, result_img, 0.4, 0.0)
    cv2.imwrite(result_img_file, result_img)
    print('Finish drawing')


def get_oriented_box_from_mask():
    result_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250414_piglet_sam')
    mask_file = os.path.join(result_dir, 'ch11_0001_masks.npy')
    contour_img_file = os.path.join(result_dir, 'ch11_0001_masks_contour.png')
    predicted_masks = np.load(mask_file)
    img_h, img_w = predicted_masks.shape[-2:]
    c = 0
    for mask in predicted_masks:
        c += 1
        if c < 5:
            continue
        mask = mask.astype(np.uint8) * 255  # mask should be binary
        mask = mask.reshape(img_h, img_w, 1)
        # find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 10 and c > 3:  # Filter small regions
                rect = cv2.minAreaRect(cnt)  # Get rotated rectangle
                box = cv2.boxPoints(rect)  # Convert to 4 corner points
                box = box.astype(np.int32)
                cv2.drawContours(mask, [box], 0, color=255, thickness=2)
        cv2.imwrite(contour_img_file, mask)
        break
    print('Finish converting')




def check_seg_mask_annotation():
    anno_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'project-5-at-2025-05-06-13-27-a8482e5c.json')
    with open(anno_file, 'r') as f:
        annotations = json.load(f)  # tasks
        task = annotations[0]  # data in the first labelling task
        objects = task['annotations'][0]['result']  # annotated boxes
        for i, obj in enumerate(objects):
            format = obj['value']['format']  # 'rle'
            values = obj['value']['rle']  #
            label = obj['value']['brushlabels']  # ['piglet 1']
            print(f'object {i} ({format}): {label}')



def test_binary2rle_mask():
    result_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250414_piglet_sam')
    mask_file = os.path.join(result_dir, 'ch11_0001_masks.npy')
    predicted_masks = np.load(mask_file)
    mask_bin = predicted_masks[0]
    mask_rle = image_utils.binary_mask_to_rle(mask_bin)
    return mask_rle



def run_sam(args):
    '''
    https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
    https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
    :param args:
    :return:
    '''
    image = args.input_image
    task_id = -1
    # img_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'ch3_250109090000_005200_0001.png')
    # anno_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'sample_annotations.json')
    img_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'ch11_20250109090000_003120_0001.png')
    anno_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'ch11_anno.json')
    result_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250414_piglet_sam')
    os.makedirs(result_dir, exist_ok=True)
    mask_file = os.path.join(result_dir, 'ch11_0001_masks.npy')
    result_img = os.path.join(result_dir, 'ch11_0001_pigsam.png')
    model = args.model
    ckpt_names = {'vit_h': 'sam_vit_h_4b8939.pth', 'vit_l': 'sam_vit_l_0b3195.pth'}
    if model not in ckpt_names.keys():
        raise ValueError(f'Model not supported yet: {model}')
    ckpt_path = os.path.join(ROOT_DIR, 'Models', ckpt_names[model])
    # sam model
    sam = sam_model_registry[model](checkpoint=ckpt_path)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    # input image
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    predictor.set_image(image)
    # load box prompts from annotation json
    frame_id = 0
    input_boxes = ls_io_utils.get_boxes_from_anno_json(anno_file, img_h, img_w, task_id)
    input_boxes = torch.tensor(input_boxes, device='cuda')
    # batched prompt inputs assume tensor and transformed
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    # output: masks, scores, logits
    # masks.shape: (number_of_masks) x H x W
    # set multimask_output as True for ambiguous prompts
    masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes,
                                          multimask_output=False)  # 13*1*1520*2688
    # output from batched input are cuda tensors
    masks = masks.cpu().numpy()
    np.save(mask_file, masks)
    print('Finish running')


def create_prediction_add_to_json():
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet')
    anno_file = os.path.join(data_dir, 'project-9-at-2025-05-19-14-18-6e02e2fa.json')
    anno_file_to_write = os.path.join(data_dir, '250519_6e02e2fa_ch1_import_test.json')
    prelabels = {}
    prelabels["data"] = {}
    prelabels["predictions"] = []

    with open(anno_file, 'r') as f:
        contents = json.load(f)  # tasks
        for cont in contents:
            for k, v in cont["data"].items():
                prelabels["data"][k] = v
            annotations = cont["annotations"]
            for anno in annotations:
                result = anno["result"]  # list
                entry = {}
                entry["result"] = result
                entry["model_version"] = 'test1'
                entry["score"] = 0.5
                prelabels["predictions"].append(entry)

    with open(anno_file_to_write, 'w') as fw:
        json.dump(prelabels, fw, indent=4)


def load_anno_run_sam_calc_obb_save_json_image():
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    img_file = os.path.join(data_dir, 'frame_00000.png')
    anno_file = os.path.join(data_dir, 'project-9-at-2025-05-19-14-18-6e02e2fa.json')
    pred_file_to_write = os.path.join(data_dir, 'ch1_predictions.json')
    mask_file_to_write = os.path.join(data_dir, 'ch1_00000_masks.npy')
    # input image
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    # load box prompts from annotation json, order is important because we need to save OBBs in the same order
    input_boxes, box_labels = ls_io_utils.get_boxes_from_anno_json(anno_file, img_h, img_w, task_id=0)
    '''
    # sam model
    model = 'vit_h'#args.model
    ckpt_names = {'vit_h': 'sam_vit_h_4b8939.pth', 'vit_l': 'sam_vit_l_0b3195.pth'}
    if model not in ckpt_names.keys():
        raise ValueError(f'Model not supported yet: {model}')
    ckpt_path = os.path.join(ROOT_DIR, 'Models', ckpt_names[model])
    sam = sam_model_registry[model](checkpoint=ckpt_path)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_boxes = torch.tensor(input_boxes, device='cuda')
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes,
                                          multimask_output=False)
    masks = masks.cpu().numpy()
    np.save(mask_file_to_write, masks)
    print(f'Finish running and saving masks at {mask_file_to_write}')
    '''
    masks = np.load(mask_file_to_write)
    # copy json entry from anno file
    prelabels = {}
    prelabels["data"] = {}
    prelabels["predictions"] = []
    with open(anno_file, 'r') as f:
        contents = json.load(f)  # tasks
        for cont in contents:
            for k, v in cont["data"].items():
                prelabels["data"][k] = v
            annotations = cont["annotations"]
            for anno in annotations:
                result = anno["result"]  # list
                entry = {}
                entry["result"] = result
                entry["model_version"] = 'test1'
                entry["score"] = 0.5
                prelabels["predictions"].append(entry)

    # calculate OBB from mask
    for i, entry in enumerate(prelabels["predictions"][0]["result"]):
        if box_labels[i] == entry["value"]["labels"][0]:
            print(box_labels[i])
            x, y, width, height, angle = ls_io_utils.get_ls_obb_from_mask(masks[i], img_w, img_h)
            prelabels["predictions"][0]["result"][i]["value"]["sequence"][0]["rotation"] = angle
            prelabels["predictions"][0]["result"][i]["value"]["sequence"][0]["x"] = x
            prelabels["predictions"][0]["result"][i]["value"]["sequence"][0]["y"] = y
            prelabels["predictions"][0]["result"][i]["value"]["sequence"][0]["width"] = width
            prelabels["predictions"][0]["result"][i]["value"]["sequence"][0]["height"] = height
        else:
            print(box_labels[i])
            print(entry["value"]["labels"][0])
            print('labels not matching')
    with open(pred_file_to_write, 'w') as fw:
        json.dump(prelabels, fw, indent=4)
    print(f'Finish calculating OBBs and saving json at {pred_file_to_write}')





def load_anno_run_sam_calc_obb_save_json_video():
    model = 'vit_l'  # 'vit_b'  # 'vit_h'  # args.model
    video_name = 'ch1_20250109090000_000100'
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    frame_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', video_name)
    anno_file = os.path.join(data_dir, 'ch1_all_abb_anno.json')  # annotations including interpolated frames
    # anno_file = os.path.join(data_dir, 'ch1_full_box_annotation.json')  # annotations for key frames only
    pred_file_to_write = os.path.join(data_dir, f'ch1_all_obb_{model}_predictions.json')
    visuals_dir = os.path.join(data_dir, 'box_mask')#'visuals')
    os.makedirs(visuals_dir, exist_ok=True)

    # sam model
    ckpt_names = {'vit_h': 'sam_vit_h_4b8939.pth', 'vit_l': 'sam_vit_l_0b3195.pth', 'vit_b': 'sam_vit_b_01ec64.pth'}
    if model not in ckpt_names.keys():
        raise ValueError(f'Model not supported yet: {model}')
    ckpt_path = os.path.join(ROOT_DIR, 'Models', ckpt_names[model])
    sam = sam_model_registry[model](checkpoint=ckpt_path)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)

    # get annotation content from json file
    prelabels = {}
    prelabels["data"] = {}
    prelabels["predictions"] = []
    f = open(anno_file, 'r')
    contents = json.load(f)
    for k, v in contents[0]["data"].items():
        prelabels["data"][k] = v
    anno_result = contents[0]["annotations"][0]["result"]
    f.close()
    entry = {}
    entry["result"] = anno_result  # copy all result into prediction
    entry["model_version"] = 'sam1'
    entry["score"] = 0.5
    prelabels["predictions"].append(entry)

    frame_ids = ls_io_utils.get_all_key_frame_id_from_video_anno(anno_result)
    print(f'Number of frames to run: {len(frame_ids)}')
    # select some frames for debugging --------------------
    visualising_frames = frame_ids  # [62, 63, 64, 65, ]#531, 1003, ] #
    for n in visualising_frames:
        # load video frame
        # input image
        img_file = os.path.join(frame_dir, f'frame_{n:04d}.png')
        if not os.path.exists(img_file):
            print(f'Image not found: {img_file}, skipping')
            continue
        result_img_file = os.path.join(visuals_dir, f'{video_name}_{model}_{n}_pen.png')
        # # select some frames for debugging --------------------
        # if n not in visualising_frames:
        #     continue
        # # select some frames for debugging --------------------
        print(f'Processing frame {n} ...')
        image_bgr = cv2.imread(img_file)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        # load box prompts from annotation
        frame_boxes, box_labels = ls_io_utils.get_frame_cv2boxes_from_video_anno(anno_result, n, img_h, img_w)
        # run sam
        predictor.set_image(image)
        input_boxes = np.array(frame_boxes, dtype=np.float32)
        input_boxes = torch.tensor(input_boxes, device='cuda')
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes,
                                              multimask_output=False)
        masks = masks.cpu().numpy()  # masks for all boxes on the frame
        result_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for i, mask in enumerate(masks):
            for j, obj in enumerate(prelabels["predictions"][0]["result"]):  # loop through all labelled objects
                pred_value = obj["value"]
                obj_label = pred_value["labels"][0]
                if box_labels[i] == obj_label:
                    for k, entry in enumerate(pred_value["sequence"]):  # loop through all key frames for the object
                        frame_id = entry['frame']
                        if frame_id == n:
                            # calculate OBB and append to prediction
                            x, y, width, height, angle = ls_io_utils.get_ls_obb_from_mask(mask, img_w, img_h)
                            prelabels["predictions"][0]["result"][j]["value"]["sequence"][k]["rotation"] = angle
                            prelabels["predictions"][0]["result"][j]["value"]["sequence"][k]["x"] = x
                            prelabels["predictions"][0]["result"][j]["value"]["sequence"][k]["y"] = y
                            prelabels["predictions"][0]["result"][j]["value"]["sequence"][k]["width"] = width
                            prelabels["predictions"][0]["result"][j]["value"]["sequence"][k]["height"] = height
                            # draw obb on image for debugging -----------------------------------
                            color = BOX_COLORS['p'+obj_label]
                            obox = ls_io_utils.get_cv2_box_from_ls_obb(x, y, width, height, angle, img_w, img_h)
                            x0, y0, x1, y1 = frame_boxes[i]
                            # mask
                            color_np = np.array(color, dtype=np.float32)
                            mask_image = mask.reshape(img_h, img_w, 1) * color_np.reshape(1, 1, -1)
                            result_img += mask_image.astype(np.uint8)
                            # input axis-aligned box
                            cv2.rectangle(result_img, (x0, y0), (x1, y1), color=color, thickness=1)
                            # calculated obb
                            cv2.drawContours(result_img, [obox], 0, color=color, thickness=3)

                            cx = int(sum(obox[:, 0]) / 4)
                            cy = int(sum(obox[:, 1]) / 4)
                            cv2.putText(result_img, 'p'+obj_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        color=(0, 0, 0), thickness=3)
                            cv2.putText(result_img, 'p'+obj_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        color=(255, 255, 255), thickness=2)
        result_img = cv2.addWeighted(image_bgr, 0.6, result_img, 0.4, 0.0)
        result_img = result_img[:, 800:2050, :]
        cv2.putText(result_img, f'frame{n}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                    color=(255, 255, 255), thickness=3)
        cv2.imwrite(result_img_file, result_img)
        # exit(0)
        # # draw obb on image for debugging -----------------------------------
    # # save predictions
    # with open(pred_file_to_write, 'w') as fw:
    #     json.dump(prelabels, fw, indent=4)
    # print(f'Finish saving json at {pred_file_to_write}')




def write_anno():
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    anno_file = os.path.join(data_dir, 'ch1_full_box_annotation.json')
    anno_file_to_write = os.path.join(data_dir, 'ch1_full_box_annotation_structure.json')
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
        with open(anno_file_to_write, 'w') as fw:
            json.dump(annotations, fw, indent=4)


def check_framescount_vs_all_frames():
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    anno_file = os.path.join(data_dir, 'ch1_all_abb_anno.json')
    f = open(anno_file, 'r')
    contents = json.load(f)
    anno_result = contents[0]["annotations"][0]["result"]
    frame_ids = ls_io_utils.get_all_key_frame_id_from_video_anno(anno_result)
    framescount = contents[0]["annotations"][0]["result"][0]["value"]["framesCount"]
    f.close()
    print(f'No. of frames: {len(frame_ids)}')
    print(f'framescount: {framescount}')


def get_abb_from_obb(anno_result, frame, img_h, img_w):
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


def copy_obbs_frame47_ch1_video():
    import copy
    model = 'vit_l'
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    pred_file = os.path.join(data_dir, f'ch1_all_obb_{model}_predictions.json')
    with open(pred_file, 'r') as f:
        contents = json.load(f)
        updated_obbs = copy.deepcopy(contents)
        result = contents[0]["predictions"][0]["result"]
        for j, obj in enumerate(result):
            pred_value = obj["value"]
            obj_label = pred_value["labels"][0]
            x, y, w, h, r = -1, -1, -1, -1, None
            for k, entry in enumerate(pred_value["sequence"]):
                frame_id = entry['frame']
                if frame_id == 48:
                    r = updated_obbs["predictions"][0]["result"][j]["value"]["sequence"][k]["rotation"]
                    x = updated_obbs["predictions"][0]["result"][j]["value"]["sequence"][k]["x"]
                    y = updated_obbs["predictions"][0]["result"][j]["value"]["sequence"][k]["y"]
                    w = updated_obbs["predictions"][0]["result"][j]["value"]["sequence"][k]["width"]
                    h = updated_obbs["predictions"][0]["result"][j]["value"]["sequence"][k]["height"]
                    break
            for fc in range(2, 48):
                seq = {}
                seq["frame"] = fc
                seq["frame"] = fc
                seq["rotation"] = fc
                seq["x"] = fc
                seq["y"] = fc
                seq["width"] = fc
                seq["height"] = fc


def verify_ls_cv2_coords():
    frame = 300
    model = 'vit_l'
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    result_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250602_ls_cv2_obb')
    os.makedirs(result_dir, exist_ok=True)
    abb_video_anno_file = os.path.join(data_dir, f'ch1_all_abb_anno.json')
    obb_video_pred_file = os.path.join(data_dir, f'ch1_all_obb_{model}_predictions.json')
    frame_dir = os.path.join(data_dir, 'images')
    img_file = os.path.join(frame_dir, f'frame_{frame:05d}.png')
    mask_file_to_write = os.path.join(result_dir, f'ch1_{frame:05d}_masks.npy')
    vis_file_to_write = os.path.join(result_dir, f'ch1_{frame:05d}_vis.png')
    pred_file_to_write = os.path.join(result_dir, f'ch1_f{frame:05d}_predictions.json')

    # get annotation content from json file
    f = open(abb_video_anno_file, 'r')
    contents = json.load(f)
    anno_result = contents[0]["annotations"][0]["result"]
    prelabels = {}
    prelabels["data"] = {}
    prelabels["predictions"] = []
    prelabels["data"]["image"] = "/data/local-files/?d=Documents/Work/QUBRF/Piglets/Data/MSUclips/sample_image/ch1_20250109090000_000100/frame_00300.png"
    all_labelled_objects = {}
    for objects in anno_result:  # loop through all labelled objects
        entry = {}
        anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
        label = 'p'+anno_value["labels"][0]  # str
        entry["original_width"] = 2688
        entry["original_height"] = 1520
        entry["image_rotation"] = 0
        # entry["id"] = result["id"]
        entry["from_name"] = "box"
        entry["to_name"] = "image"
        entry["type"] = "RectangleLabels"
        entry["origin"] = "automatic"
        entry["value"] = {}  # x,y,width,height, rotation,labels
        for frame_anno in anno_value["sequence"]:  # loop through all labelled key frames
            frame_id = frame_anno['frame']
            if frame_id == frame:
                entry["value"]['x'] = frame_anno['x']
                entry["value"]['y'] = frame_anno['y']
                entry["value"]['width'] = frame_anno['width']
                entry["value"]['height'] = frame_anno['height']
                entry["value"]['rotation'] = frame_anno['rotation']
                entry["value"]['labels'] = [label]
                # break
        entry["model_version"] = 'sam1'
        entry["score"] = 0.5
        all_labelled_objects
    # for result in anno_result:
    #     for attributes, values in result.items():
    #         if attributes == 'value': ##isinstance(values, dict):
    #             entry[attributes] ={}
    #             for k, v in values.items():
    #                 entry[attributes][k] = v
    #         else:
    #             entry[attributes] = values
    # entry["result"] = anno_result

    prelabels["predictions"].append(entry)
    f.close()
    with open(pred_file_to_write, 'w') as outfile:
        json.dump(prelabels, outfile, indent=4)
    exit(0)
    image_bgr = cv2.imread(img_file)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    input_boxes, box_labels = ls_io_utils.get_frame_cv2boxes_from_video_anno(anno_result, frame, img_h, img_w)

    # # sam model
    # ckpt_names = {'vit_h': 'sam_vit_h_4b8939.pth', 'vit_l': 'sam_vit_l_0b3195.pth', 'vit_b': 'sam_vit_b_01ec64.pth'}
    # if model not in ckpt_names.keys():
    #     raise ValueError(f'Model not supported yet: {model}')
    # ckpt_path = os.path.join(ROOT_DIR, 'Models', ckpt_names[model])
    # sam = sam_model_registry[model](checkpoint=ckpt_path)
    # sam.to(device='cuda')
    # predictor = SamPredictor(sam)
    # predictor.set_image(image)
    # input_boxes = torch.tensor(input_boxes, device='cuda')
    # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    # masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes,
    #                                       multimask_output=False)
    # masks = masks.cpu().numpy()
    # np.save(mask_file_to_write, masks)
    # print(f'Finish running and saving masks at {mask_file_to_write}')

    masks = np.load(mask_file_to_write)
    mask_image = np.zeros_like(image)
    oboxes, olabels = [], []
    for i, (abb, ablb) in enumerate(zip(input_boxes, box_labels)):
        x0, y0, x1, y1 = abb
        label = 'p' + ablb
        color = BOX_COLORS[label]
        cv2.rectangle(image_bgr, (x0, y0), (x1, y1), color=color, thickness=2)
    # for mask in masks:
        mask = masks[i]
        color_np = np.array(color, dtype=np.float32)
        mask_obj = mask.reshape(img_h, img_w, 1) * color_np.reshape(1, 1, -1)
        mask_image += mask_obj.astype(np.uint8)
        cx, cy, obw, obh, obr = ls_io_utils.calc_obb_from_mask_cv2(mask, img_w, img_h)
        oboxes.append([cx, cy, obw, obh, obr])
        olabels.append(label)
        obox = cv2.boxPoints(((cx, cy), (obw, obh), obr))
        obox = obox.astype(np.int32)
        cv2.drawContours(image_bgr, [obox], 0, color=color, thickness=2)
        cx = int(sum(obox[:, 0]) / 4)
        cy = int(sum(obox[:, 1]) / 4)
        cv2.putText(image_bgr, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 0, 0), thickness=3)
        cv2.putText(image_bgr, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(255, 255, 255), thickness=2)
    # result_img = cv2.addWeighted(image_bgr, 0.7, mask_image, 0.3, 0.0)
    # result_img = result_img[:, 800:2050, :]
    # cv2.imwrite(vis_file_to_write, result_img)
    # np.savetxt(os.path.join(result_dir, f'f{frame}_obbs.txt'), oboxes, delimiter=',', fmt='%d', header='')
    # np.savetxt(os.path.join(result_dir, f'f{frame}_olabels.txt'), olabels, fmt='%s', header='')


    # obbs = np.loadtxt(os.path.join(kp_dir, f'f{n}_obbs.txt'), delimiter=',', dtype=np.int32)
    # obb_labels = np.loadtxt(os.path.join(kp_dir, f'f{n}_obb_labels.txt'), dtype=str)


    # with open(obb_video_pred_file, 'r') as f:
    #     contents = json.load(f)
    #     pred_result = contents["predictions"][0]["result"]
    #     for objects in pred_result:  # loop through all labelled objects
    #         anno_value = objects["value"]  # boxes and labels of a specific object in all key frames
    #         label = anno_value["labels"][0]  # str
    #         for entry in anno_value["sequence"]:  # loop through all labelled key frames
    #             frame_id = entry['frame']
    #             if frame_id == frame:  # be careful about frame ID indexing
    #                 angle = int(entry['rotation'] - 180)
    #                 cx = entry['x'] * img_w / 100.
    #                 cy = entry['y'] * img_h / 100.
    #                 w = int(entry['width'] * img_w / 100.)
    #                 h = int(entry['height'] * img_h / 100.)
    #                 cx = int(cx + w / 2.)
    #                 cy = int(cy + h / 2.)
    #                 cv2.drawContours(result_img, [obox], 0, color=color, thickness=3)


def structure_ls_export_json():
    # data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'video_obb_annotations')
    in_json_file = os.path.join(data_dir, 'project-15-at-2025-06-10-13-32-e82052f6.json')
    # out_json_file = os.path.join(data_dir, 'project15-250609-video-obb-structured.json')
    # ls_io_utils.write_ls_export_json_into_structured(in_json_file, out_json_file)
    ls_io_utils.separate_annotation_per_video(in_json_file)


def lsobb_to_lsabb():
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'anno_test')
    anno_json_file = os.path.join(data_dir, 'project14-250604-image-obb-structured.json')
    with open(anno_json_file, 'r') as f:
        data = json.load(f)

    image  =cv2.imread(os.path.join(data_dir, 'project14-250604'))
    # Process each task
    for task in data:
        predictions = []
        image_width = task['annotations'][0]['result'][0]['original_width']
        image_height = task['annotations'][0]['result'][0]['original_height']

        for item in task['annotations'][0]['result']:
            val = item['value']
            obox, abox = ls_io_utils.calc_cv2abb_from_lsobb(val['x'], val['y'], val['width'], val['height'], image_width, image_height)

            # LS OBB in percentile format, Convert to pixels
            box_w = val['width'] / 100 * image_width
            box_h = val['height'] / 100 * image_height
            top_left_x = val['x'] / 100 * image_width
            top_left_y = val['y'] / 100 * image_height
            angle_deg = val['rotation']
            angle_rad = math.radians(angle_deg)
            # Center of box
            cx = top_left_x + box_w / 2
            cy = top_left_y + box_h / 2

            # # Corners relative to center
            # corners = [
            #     (-box_w / 2, -box_h / 2),
            #     (box_w / 2, -box_h / 2),
            #     (box_w / 2, box_h / 2),
            #     (-box_w / 2, box_h / 2),
            # ]
            # # Rotate corners
            # rotated_corners = []
            # for dx, dy in corners:
            #     rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            #     ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            #     rotated_corners.append((cx + rx, cy + ry))
            #
            # xs = [p[0] for p in rotated_corners]
            # ys = [p[1] for p in rotated_corners]
            # x_min, y_min = min(xs), min(ys)
            # x_max, y_max = max(xs), max(ys)
            #
            # # Convert back to percentages
            # new_x = x_min / image_width * 100
            # new_y = y_min / image_height * 100
            # new_w = (x_max - x_min) / image_width * 100
            # new_h = (y_max - y_min) / image_height * 100


def ls_json_2_mot_csv():
    spec_frame = 250  # 900  # 500  # 1300  # 300
    video_name = 'ch12_20250109090000_003000'
    img_w, img_h = 2688, 1520
    data_dir = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'video_obb_annotations')
    anno_json_file = os.path.join(data_dir, f'{video_name}.json')
    # out_csv_file = os.path.join(data_dir, f'{video_name}.txt')
    vis_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250617_lsobb2_mot')
    out_csv_file = os.path.join(vis_dir, f'{video_name}_f{spec_frame}.txt')
    # ls_io_utils.write_ls_video_export_json_into_mot_csv(anno_json_file, out_csv_file, img_w, img_h, obj=-1, max_frame=2)
    ls_io_utils.write_ls_video_export_json_into_mot_csv(anno_json_file, out_csv_file, img_w, img_h,
                                                        obj=-1, max_frame=-1, spec_frame=spec_frame)

import csv
def draw_verify_mot_box():
    frame_number = 300 # 900  # 500  # 1300  # 1
    anno_frame = 250
    video_name = 'ch12_20250109090000_003000'
    image_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'Raw', f'{video_name}_{frame_number:04d}.png')
    # csv_file = os.path.join(ROOT_DIR, 'WTData', 'Piglet', 'video_obb_annotations', f'{video_name}.txt')
    vis_dir = os.path.join(ROOT_DIR, 'WTData', 'Results', '250617_lsobb2_mot')
    csv_file = os.path.join(vis_dir, f'{video_name}_f{anno_frame}.txt')
    os.makedirs(vis_dir, exist_ok=True)
    vis_file = os.path.join(vis_dir, f'{video_name}_f{frame_number}_mot_box.png')
    image = cv2.imread(image_file)
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', )
        for row in reader:
            frame = int(row[0])
            id = int(row[1])
            # x0 = int(row[2])
            # y0 = int(row[3])
            # x1 = x0 + int(row[4])
            # y1 = y0 + int(row[5])
            if frame == anno_frame:#frame_number:
                color = BOX_COLORS['p'+str(id)]
                # print(f'frame {frame_number} obj {id} {x0} {y0} {x1} {y1}')
                # cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=3)
                # cv2.putText(image, str(id), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             color=(255, 255, 255), thickness=3)
                # cv2.putText(image, str(id), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                #             thickness=2)
                x0, y0 = -1, -1
                for k in range(2, 10, 2):
                    x, y = int(float(row[k])), int(float(row[k+1]))
                    print(f'frame {frame_number} obj {id} {x} {y}')
                    cv2.circle(image, (x, y), radius=10, color=color, thickness=3)
                    if x0 > 0 and y0 > 0:
                        cv2.line(image, (x0, y0), (x, y), color=color, thickness=2)
                    cv2.putText(image, str(k), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                color=color, thickness=3)
                    x0, y0 = x, y
    # image = image[:, 500:1900, :]
    cv2.putText(image, f'frame {frame_number}', (50, 1500), cv2.FONT_HERSHEY_SIMPLEX, 2,
                color=(255, 255, 255), thickness=3)
    cv2.imwrite(vis_file, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SAM with box prompts')
    parser.add_argument('--input_image', type=str)
    parser.add_argument('--model', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    args = parser.parse_args()

    # get_boxes_from_anno_json()
    # run_sam(args)
    # convert_mask_to_image()
    # overlay_image_mask_for_visual()
    # draw_box_mask_for_visual()
    # get_oriented_box_from_masks()
    # check_seg_mask_annotation()
    # create_prediction_add_to_json()
    # load_anno_run_sam_calc_obb_save_json_image()
    # write_anno()
    # load_anno_run_sam_calc_obb_save_json_video()
    # check_framescount_vs_all_frames()
    # verify_ls_cv2_coords()
    # structure_ls_export_json()
    ls_json_2_mot_csv()
    draw_verify_mot_box()