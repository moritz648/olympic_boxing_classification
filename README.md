# PKI

To setup and test this code you need to do the following steps:
1. python3 -m venv env
2. source env/bin/activate
3. Segmentation of fighters:
   1. cd segmentation_and_training
   2. pip install -U pip && pip install -r requirements.txt
   3. gdown 1M7INrxU2h5fCPgSCPzpt1cw_XdhQ8VHE
   4. cd ../demo && python masking.py
   5. A window should open showing a video of the fight, press/hold x to skip frames until both fighters are shown in their red/blue outfits. Then, select the head & body of each of the fighters as well as the referee (in this order). After selecting, press s to start the segmentation process which outputs a JSON file with the bounding boxes of the tracked fighters
4. Pose Estimation
   1. cd ../../pose_estimation
   2. pip install --upgrade setuptools && pip install -r requirements.txt
   3. gdown 1uiJAfE4x_XxFDvBKbku91hWpr6QcRttI
   4. python loaded_bbox.py (processes the boundingboxes by doing pose estimation on the fighters and returning a JSON file of the keypoints)
5. Training
  1. To do training we first download a small sample of processed data: cd ../segmentation_and_training && gdown 1O2oFlZu9tbkrzSN7du5d1JCvIZB3plDK && unzip data.zip -d 'Olympic Boxing Punch Classification Video Dataset'
  2. We can then either train the transformer: cd demo/transformer && python train_transformer_2.py or finetune the LLM: cd demo/llm_sentenceification && python finetune_llm.py
