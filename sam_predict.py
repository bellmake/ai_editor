from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob 
import os

from sam_utils import show_box, show_mask, show_points


sam = sam_model_registry["vit_b"](checkpoint="/home/joseph/study/multimodal/ai_editor/sam_vit_b_01ec64.pth")
sam = sam.cuda()

predictor = SamPredictor(sam)


# sample image (shushi)
# fname = '/home/joseph/study/multimodal/ai_editor/my_data/optimus2.png'
fname = '/home/joseph/study/multimodal/ai_editor/my_data/livingroom.png'

img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predictor.set_image(img)

print(predictor.features.shape)

input_point = np.array([[1100, 200],[1100, 300]])  # points

input_label = np.array([1,1])  # 1: foreground / 0: background

plt.figure(figsize=(10,10))
plt.imshow(img)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.savefig('/home/joseph/study/multimodal/ai_editor/my_data_result/point2.png')
plt.close()

masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                        )

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig('point.png')
    plt.close()
    binary_array = mask.astype(np.uint8)
    cv2.imwrite('/home/joseph/study/multimodal/ai_editor/my_data/mask.png',binary_array*255)


# input_point = np.array([[200,50]])  # points

# input_label = np.array([1])  # 1: foreground / 0: background

# plt.figure(figsize=(10,10))
# plt.imshow(img)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.savefig('/home/joseph/study/multimodal/ai_editor/point.png')
# plt.close()

# masks, scores, logits = predictor.predict(
#                         point_coords=input_point,
#                         point_labels=input_label,
#                         multimask_output=True,
#                         )

# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(img)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.savefig('point.png')
#     plt.close()


# input_point = np.array([[200, 180],[190,90]])  # points

# input_label = np.array([1,0])  # 1: foreground / 0: background

# plt.figure(figsize=(10,10))
# plt.imshow(img)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.savefig('/home/joseph/study/multimodal/ai_editor/point.png')
# plt.close()

# masks, scores, logits = predictor.predict(
#                         point_coords=input_point,
#                         point_labels=input_label,
#                         multimask_output=True,
#                         )

# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(img)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.savefig('point.png')
#     plt.close()