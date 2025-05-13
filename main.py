from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# モデル読み込み
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# 画像読み込み（ファイル名を調整）
image_path = "fashion_image.jpg"
image = Image.open(image_path).convert("RGB")

# 分類候補テキスト
texts = [
    "y2k fashion", "techwear", "normcore", "korean casual", "90s vintage",
    "monochrome outfit", "colorful streetwear", "oversized silhouette"
]

# 前処理
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 推論
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)[0]

# 結果表示
for text, prob in zip(texts, probs):
    print(f"{text}: {prob.item():.4f}")
