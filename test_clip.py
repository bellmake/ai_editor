from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


txt = ['a piece of sushi','a dog', 'a robot']
img = Image.open('/home/joseph/study/multimodal/ai_editor/my_data/optimus.jpg')

inputs = processor(text=txt, images=img, return_tensors="pt", padding=True)
print(inputs.keys())

outputs= model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)

# 가장 높은 확률을 가지는 텍스트의 인덱스 추출
max_prob_index = torch.argmax(probs).item()

# 가장 높은 확률을 가지는 텍스트 출력
print(f"The most relevant text is: {txt[max_prob_index]}")