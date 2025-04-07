import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

pose_config = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
pose_model = init_model(pose_config, pose_checkpoint, device='cpu')
base_path = '../segmentation_and_training/demo/example_data'

MIN_BBOX_AREA = 5000

def load_bounding_boxes(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_bounding_boxes_for_frame(bounding_boxes, frame_index):
    for frame_data in bounding_boxes:
        if frame_data["frame"] == frame_index:
            return frame_data["boxes"]
    return []

def filter_small_bounding_boxes(bounding_boxes):
    return [box for box in bounding_boxes if box['w'] * box['h'] >= MIN_BBOX_AREA]

def draw_pose(image, pose_results):
    if not pose_results:
        return image, None
    keypoints = []
    for person in pose_results:
        kp = person.pred_instances.keypoints
        if kp is None or len(kp) == 0:
            continue

        for point in kp[0]:
            if len(point) == 2:
                x, y = point
                keypoints.append([x, y])
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    return image, keypoints


def process_frame_with_bounding_boxes(image, bounding_boxes):
    bounding_boxes = filter_small_bounding_boxes(bounding_boxes)

    if not bounding_boxes:
        return image, None

    color_map = {1: (0, 0, 255), 2: (0, 255, 0)}
    person_bboxes = []
    keypoints = {}

    for box in bounding_boxes:
        x1, y1, w, h = box['x'], box['y'], box['w'], box['h']
        x2, y2 = x1 + w, y1 + h
        person_bboxes.append([x1, y1, x2, y2])

        obj_id = box.get('object_id', 0)
        color = color_map.get(obj_id, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        pose_results = inference_topdown(
            pose_model,
            image,
            bboxes=np.array(person_bboxes),
            bbox_format='xyxy'
        )

        image, person_keypoints = draw_pose(image, pose_results)
        keypoints[obj_id] = person_keypoints
        person_bboxes = []

    return image, keypoints


def detect_and_estimate_pose_with_boxes(input_path, bbox_file, output_path):
    bounding_boxes = load_bounding_boxes(bbox_file)
    result_keypoints = []

    if os.path.splitext(input_path)[-1].lower() in ['.mp4', '.avi', '.mov']:
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_w, new_h = int(orig_w * 0.5), int(orig_h * 0.5)  # Scale down by 50%

        out = cv2.VideoWriter(os.path.join(output_path, 'output_filtered.mp4'), fourcc, cap.get(cv2.CAP_PROP_FPS), (new_w, new_h))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=frame_count, desc="Processing Frames with Bounding Boxes", unit="frame") as pbar:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Resize frame
                frame_boxes = get_bounding_boxes_for_frame(bounding_boxes, frame_index)

                processed_frame, keypoints = process_frame_with_bounding_boxes(frame, frame_boxes)
                result_keypoints.append({"frame": frame_index, "keypoints": keypoints})
                out.write(processed_frame)
                pbar.update(1)
                frame_index += 1

        cap.release()
        out.release()
        print("Processed video saved as 'output_filtered.mp4'")
        return result_keypoints
    else:
        image = cv2.imread(input_path)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame_boxes = get_bounding_boxes_for_frame(bounding_boxes, 0)

        output_image, _ = process_frame_with_bounding_boxes(image, frame_boxes)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        cv2.imwrite('output_image_filtered.png', output_image)
        print("Processed image saved as 'output_image_filtered.png'")



def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder)
    bbox_file = None
    video_file = None
    for file in os.listdir(os.path.join(full_path, 'data')):
        if file.endswith('.json'):
            bbox_file = os.path.join(full_path, 'data', file)
        elif file.endswith('.mp4'):
            video_file = os.path.join(full_path, 'data', file)
        if bbox_file and video_file:
            result_keypoints = detect_and_estimate_pose_with_boxes(video_file, bbox_file, full_path)
            with open(os.path.join(full_path, 'data', "keypoints.json"), "w") as f:
                json.dump(convert_to_serializable(result_keypoints), f, indent=4)