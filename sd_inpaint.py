from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from PIL import ImageOps

import torch
import numpy as np
import cv2

pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting',
                                                    revision='fp16',
                                                    torch_dtype=torch.float16)

pipe = pipe.to('cuda')
prompt = 'a window with blue ocean scenary'
image = Image.open('/home/joseph/study/multimodal/ai_editor/my_data/livingroom.png')
image = ImageOps.exif_transpose(image)
image = image.resize((512,512))
mask_image = Image.open('/home/joseph/study/multimodal/ai_editor/my_data/mask.png')

kernel = np.ones((3,3), np.uint8)
mask_image = cv2.dilate(np.array(mask_image), kernel, iterations=10) # 경계영역 넓혀줌
mask_image = cv2.resize(mask_image, (512,512))

# mask_image의 채널 수가 3일 경우(컬러 이미지) 흑백으로 변환
if mask_image.ndim == 3:
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

image = pipe(prompt=prompt, image=image, mask_image=Image.fromarray(mask_image)).images[0]

image.save('/home/joseph/study/multimodal/ai_editor/my_data_result/inpainted.png')