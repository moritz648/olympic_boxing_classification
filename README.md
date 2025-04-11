# PKI

To set up and test this code, follow these steps:

## 1. Create a Python Virtual Environment

Run the following command in your terminal:

```bash
python3 -m venv env
```

## 2. Activate the Virtual Environment

Activate the virtual environment with:

```bash
source env/bin/activate
```

## 3. Install CUDA Toolkit
```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

## 4. Segmentation of Fighters

1. **Navigate to the segmentation and training directory:**

       cd segmentation_and_training

2. **Install the required packages:**

       pip install -U pip && pip install -r requirements.txt

3. **Download the required file:**

       gdown 1M7INrxU2h5fCPgSCPzpt1cw_XdhQ8VHE

4. **Run the segmentation demo:**

       cd ../demo && python masking.py

5. **Follow the on-screen instructions:**
   - A window should open showing a video of the fight.
   - Press/hold `x` to skip frames until both fighters are shown in their red/blue outfits.
   - Then, select the head & body of each fighter as well as the referee (in this order).
   - After selecting, press `s` to start the segmentation process. This outputs a JSON file with the bounding boxes of the tracked fighters.

## 5. Pose Estimation

1. **Navigate to the pose estimation directory:**

       cd ../../pose_estimation

2. **Upgrade setuptools and install the necessary packages:**

       pip install --upgrade setuptools && pip install -r requirements.txt

3. **Download the required file:**

       gdown 1uiJAfE4x_XxFDvBKbku91hWpr6QcRttI

4. **Run the bounding box processor:**

       python loaded_bbox.py

   This processes the bounding boxes by performing pose estimation on the fighters and returns a JSON file with the keypoints.

## 6. Training

1. **Download a small sample of processed data:**

       cd ../segmentation_and_training
       gdown 1O2oFlZu9tbkrzSN7du5d1JCvIZB3plDK
       unzip data.zip -d 'Olympic Boxing Punch Classification Video Dataset'

2. **Choose your training option:**

   - **Train the transformer:**

         cd demo/transformer && python train_transformer_2.py

   - **Fine-tune the LLM:**

         cd demo/llm_sentenceification && python finetune_llm.py
