import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

pose_config = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
pose_model = init_model(pose_config, pose_checkpoint, device='cpu')

det_config = 'mmdetection/rtmdet_tiny_8xb32-300e_coco.py'
det_checkpoint = 'mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
det_model = init_detector(det_config, det_checkpoint, device='cpu')

det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)

def filter_person_detections(mmdet_results, score_thr=0.5):
    pred_instances = mmdet_results.pred_instances

    bboxes = pred_instances.bboxes.numpy()
    scores = pred_instances.scores.numpy()
    labels = pred_instances.labels.numpy()

    person_bboxes = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if label == 0 and score > score_thr:
            person_bboxes.append(bbox)
    return np.array(person_bboxes)

def draw_pose(image, pose_results):
    if not pose_results:
        print("No pose results found.")
        return image

    for person in pose_results:
        keypoints = person.pred_instances.keypoints
        if keypoints is None or len(keypoints) == 0:
            print("No keypoints found for this person.")
            continue

        for kp in keypoints[0]:
            if len(kp) == 2:
                x, y = kp
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
    return image

def process_frame(image):
    mmdet_results = inference_detector(det_model, image)
    person_bboxes = filter_person_detections(mmdet_results)

    if person_bboxes.size == 0:
        print("No persons detected.")
        return image

    pose_results = inference_topdown(
        pose_model,
        image,
        bboxes=person_bboxes,
        bbox_format='xyxy'
    )

    for bbox in person_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    image = draw_pose(image, pose_results)
    return image

def detect_and_estimate_pose(input_path):
    if os.path.splitext(input_path)[-1].lower() in ['.mp4', '.avi', '.mov']:
        # Process video
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=frame_count, desc="Processing Frames", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame(frame)
                out.write(processed_frame)
                pbar.update(1)

        cap.release()
        out.release()
        print("Processed video saved as 'output.mp4'")
    else:
        image = cv2.imread(input_path)
        output_image = process_frame(image)

        output_image = cv2.resize(output_image, dsize=None, fx=0.5, fy=0.5)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        cv2.imwrite('output_image.png', output_image)
        print("Processed image saved as 'output_image.png'")

input_path = 'test.mp4'
detect_and_estimate_pose(input_path)