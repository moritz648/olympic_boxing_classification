import json
import os.path
import cv2

base_path = r'C:\Users\willi\PycharmProjects\segment-anything-2-real-time\Olympic Boxing Punch Classification Video Dataset\task_kam2_gh078416'

with open(os.path.join(base_path, "annotations.json"), "r") as f:
    annotations = json.load(f)

annotations_dict = {}
for track in annotations[0]['tracks']:
    label = track["label"]
    for shape in track["shapes"]:
        frame_num = shape["frame"]
        points = shape["points"]
        box = [int(points[0]), int(points[1]), int(points[2]), int(points[3])]
        if frame_num not in annotations_dict:
            annotations_dict[frame_num] = []
        annotations_dict[frame_num].append((box, label))

with open(r"C:\Users\willi\PycharmProjects\segment-anything-2-real-time\Olympic Boxing Punch Classification Video Dataset\task_kam2_gh078416\data\keypoints.json", "r") as f:
    keypoints_data = json.load(f)

keypoints_dict = {}
for entry in keypoints_data:
    frame_num = entry["frame"]
    keypoints_dict[frame_num] = entry["keypoints"]

video_path = os.path.join(base_path, "data", "GH078416.mp4")
cap = cv2.VideoCapture(video_path)

frame_num = 0
scale_factor = 0.5
while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_w = int(frame.shape[1] * scale_factor)
    new_h = int(frame.shape[0] * scale_factor)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    frame_num += 1

    if frame_num in annotations_dict:
        for box, label in annotations_dict[frame_num]:
            x1 = int(box[0] * scale_factor)
            y1 = int(box[1] * scale_factor)
            x2 = int(box[2] * scale_factor)
            y2 = int(box[3] * scale_factor)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    if frame_num in keypoints_dict:
        kps = keypoints_dict[frame_num]
        if kps is not None:
            fighter_colors = {
                "1": (255, 0, 0),
                "2": (0, 0, 255)
            }
            for fighter_id, points in kps.items():
                color = fighter_colors.get(fighter_id, (0, 255, 255))
                for pt in points:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, color, -1)
                if points:
                    pt0 = points[0]
                    cv2.putText(frame, f"KP {fighter_id}", (int(pt0[0]), int(pt0[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Video", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
