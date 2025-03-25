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
        return image

    for person in pose_results:
        print(person)
        keypoints = person.pred_instances.keypoints
        if keypoints is None or len(keypoints) == 0:
            continue

        for kp in keypoints[0]:
            if len(kp) == 2:
                x, y = kp
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    return image

def process_frame_with_bounding_boxes(image, bounding_boxes):
    bounding_boxes = filter_small_bounding_boxes(bounding_boxes)

    if not bounding_boxes:
        return image

    color_map = {1: (0, 0, 255), 2: (0, 255, 0)}
    person_bboxes = []

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

    image = draw_pose(image, pose_results)
    return image

def detect_and_estimate_pose_with_boxes(input_path, bbox_file):
    bounding_boxes = load_bounding_boxes(bbox_file)

    if os.path.splitext(input_path)[-1].lower() in ['.mp4', '.avi', '.mov']:
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_filtered.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=frame_count, desc="Processing Frames with Bounding Boxes", unit="frame") as pbar:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_boxes = get_bounding_boxes_for_frame(bounding_boxes, frame_index)
                processed_frame = process_frame_with_bounding_boxes(frame, frame_boxes)
                out.write(processed_frame)
                pbar.update(1)
                frame_index += 1

        cap.release()
        out.release()
        print("Processed video saved as 'output_filtered.mp4'")
    else:
        image = cv2.imread(input_path)
        frame_boxes = get_bounding_boxes_for_frame(bounding_boxes, 0)
        output_image = process_frame_with_bounding_boxes(image, frame_boxes)

        output_image = cv2.resize(output_image, dsize=None, fx=0.5, fy=0.5)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        cv2.imwrite('output_image_filtered.png', output_image)
        print("Processed image saved as 'output_image_filtered.png'")

input_path = 'test.mp4'
bbox_file = 'bounding_boxes.json'
detect_and_estimate_pose_with_boxes(input_path, bbox_file)
