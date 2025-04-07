import json
import os
import cv2
import numpy as np
import mediapipe as mp
import torch
from sam2.build_sam import build_sam2_camera_predictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def compare_masks(mask1: np.ndarray, mask2: np.ndarray, overlap_threshold: float = 0.99) -> bool:
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return False
    return (intersection / union) >= overlap_threshold

base_path = 'example_data'

selected_points = []
current_object_id = 1

def select_point(event, x, y, flags, param):
    global selected_points, current_object_id
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 6:
        selected_points.append((x, y, current_object_id))
        print(f"Point selected for fighter {current_object_id}: {(x, y)}")
        if len(selected_points) % 2 == 0 and len(selected_points) < 6:
            current_object_id += 1

for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder)
    #if 'bounding_boxes.json' in os.listdir(os.path.join(full_path, 'data')):
    #    continue
    for video in os.listdir(os.path.join(full_path, 'data')):
        if not video.endswith('.mp4'):
            continue
        print(f"Processing video: {video}")
        cap = cv2.VideoCapture(os.path.join(full_path, 'data', video))
        frame_count = -1
        selected_points = []
        current_object_id = 1
        skipping = True

        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", select_point)

        while skipping:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Failed to read video.")
            new_w = int(frame.shape[1] * 0.5)
            new_h = int(frame.shape[0] * 0.5)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", frame_rgb)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("x"):
                continue
            elif key == ord("s"):
                skipping = False

        predictor.load_first_frame(frame_rgb)
        for obj_id in [1, 2]:
            pts = np.array([p[:2] for p in selected_points if p[2] == obj_id], dtype=np.float32)
            labels = np.ones(len(pts), dtype=np.int32)
            predictor.add_new_prompt(frame_idx=0, obj_id=obj_id, points=pts, labels=labels)

        bounding_boxes = []
        fighter_masks_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            new_w = int(frame.shape[1] * 0.5)
            new_h = int(frame.shape[0] * 0.5)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

            fighter_masks = []
            referee_box = None
            temp_boxes = []

            for i, obj_id in enumerate(out_obj_ids):
                mask_logits = out_mask_logits[i].detach().cpu()
                mask = ((mask_logits > 0).numpy().squeeze().astype(np.uint8)) * 255
                x, y, w, h = cv2.boundingRect(mask)
                if obj_id == 3:
                    referee_box = (x, y, w, h)
                else:
                    temp_boxes.append((obj_id, x, y, w, h, mask))
                    fighter_masks.append(mask)

            need_reselect = False
            if len(fighter_masks) != 2:
                need_reselect = True
            elif compare_masks(fighter_masks[0], fighter_masks[1], overlap_threshold=0.99):
                need_reselect = True

            if need_reselect:
                print("Segmentation error: Duplicate mask for one fighter or missing mask detected. Reselect prompts.")
                selected_points = []
                current_object_id = 1
                cv2.setMouseCallback("frame", select_point)
                while len(selected_points) < 4:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("s"):
                        break
                frame_count = 0
                predictor.load_first_frame(frame_rgb)
                for obj_id in [1, 2]:
                    pts = np.array([p[:2] for p in selected_points if p[2] == obj_id], dtype=np.float32)
                    labels = np.ones(len(pts), dtype=np.int32)
                    predictor.add_new_prompt(frame_idx=frame_count, obj_id=obj_id, points=pts, labels=labels)
                continue
            frame_boxes = []
            colors = [(0, 255, 0), (255, 0, 0)]
            mask_overlay = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

            frame_masks = []

            for obj_id, x, y, w, h, mask in temp_boxes:
                if referee_box is not None:
                    ref_x, ref_y, ref_w, ref_h = referee_box
                    overlap = not (x + w <= ref_x or ref_x + ref_w <= x or y + h <= ref_y or ref_y + ref_h <= y)
                    cv2.circle(frame, (x, y + h), 3, colors[obj_id], thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (ref_x, ref_y + ref_h), 3, colors[obj_id], thickness=-1, lineType=cv2.LINE_AA)
                    if overlap and ref_y + ref_h > y + h:
                        continue
                frame_boxes.append({"object_id": int(obj_id), "x": x, "y": y, "w": w, "h": h})
                frame_masks.append({"object_id": int(obj_id), "mask": mask.tolist()})
                #cv2.rectangle(frame, (x, y), (x + w, y + h), colors[obj_id - 1], 2)
                mask_colored = np.zeros_like(mask_overlay)
                mask_colored[:, :, 0] = mask * (colors[obj_id - 1][2] // 255)
                mask_colored[:, :, 1] = mask * (colors[obj_id - 1][1] // 255)
                mask_colored[:, :, 2] = mask * (colors[obj_id - 1][0] // 255)
                mask_overlay = cv2.add(mask_overlay, mask_colored)

            bounding_boxes.append({"frame": frame_count, "boxes": frame_boxes})
            fighter_masks_data.append({"frame": frame_count, "masks": frame_masks})
            frame_with_mask = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)
            frame_with_mask_bgr = cv2.cvtColor(frame_with_mask, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", frame_with_mask_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        #with open(os.path.join(full_path, 'data', "bounding_boxes.json"), "w") as f:
        #    json.dump(bounding_boxes, f, indent=4)
        with open(os.path.join(full_path, 'data', "fighter_masks.json"), "w") as f:
            json.dump(fighter_masks_data, f, indent=4)
        cap.release()
        cv2.destroyAllWindows()
