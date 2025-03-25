# PKI

- Folder segmentation and training contains the segmentation approach with a requirements.txt to install the required libraries. The code was tested on python 3.12
  - run masking.py and click on the body and head of each fighter as well as the referee (the referee has to be last)
  - masking.py then does the segmentation and creates a JSON file with the bounding boxes as well as the object ids (id 1 = fighter 1, id 2 = fighter 2, id 3 = referee)
-  Folder pose_estimation contains the pose estimation approach
  - loaded_bbox.py uses the created JSON file to do the pose estimation and returns a video with the keypoints of the fighter

conda environment for running the mmaction part
![image](https://github.com/user-attachments/assets/71823a4f-ffac-4a53-9f7d-7997c1bd75e7)
