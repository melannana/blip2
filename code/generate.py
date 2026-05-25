import os
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import random
import torch
from PIL import Image
from transformers import CLIPProcessor
from model import MiniBLIP2
import pandas as pd

def generate_examples():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = MiniBLIP2(device=device).to(device)
    model.load_state_dict(torch.load("./outputs/mini_blip2_best.pt", map_location=device))
    model.eval()
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    data_root = "./data/flickr8k"
    df = pd.read_csv(os.path.join(data_root, 'captions.txt'))
    
    # 获取前 200 张唯一图片名（与训练所用的图片集合一致）
    unique_images = df['image'].unique()[:200]
    # 从中随机抽取 4 张（可固定随机种子以保证可复现，这里不固定体现“随机”）
    test_images = random.sample(list(unique_images), 4)
    print(f"随机选取的测试图片: {test_images}")
    
    results = []
    for img_name in test_images:
        img_path = os.path.join(data_root, 'Images', img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue
        image = Image.open(img_path).convert('RGB')
        pixel_values = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
        
        with torch.no_grad():
            generated_caption = model.generate(pixel_values)[0]
        
        # 取第一条真实 caption 作为参考（每个图片有5条标注）
        true_caption = df[df['image'] == img_name]['caption'].iloc[0]
        results.append((img_name, true_caption, generated_caption))
        print(f"Image: {img_name}\n  True: {true_caption}\n  Pred: {generated_caption}\n")
    
    os.makedirs("./report", exist_ok=True)
    with open("./report/generation_results.txt", "w") as f:
        for img_name, true_cap, pred_cap in results:
            f.write(f"{img_name}\nTrue: {true_cap}\nPred: {pred_cap}\n\n")

if __name__ == "__main__":
    generate_examples()