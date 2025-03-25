import cv2
import matplotlib.pyplot as plt
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or 'cuda:0'

image_path = 'test1.png'
results = inference_topdown(model, image_path)

def draw_keypoints(image_path, results):
    image = cv2.imread(image_path)
    for sample in results:
        keypoints = sample.pred_instances.keypoints
        for kp in keypoints:
            if kp.shape[-1] == 3:
                for x, y, score in kp:
                    if score > 0.5:
                        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            elif kp.shape[-1] == 2:
                for x, y in kp:
                    cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
    return image

output_image = draw_keypoints(image_path, results)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
