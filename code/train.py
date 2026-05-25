import os
# 设置镜像，加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import Flickr8kMini
from model import MiniBLIP2

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ---------- 修改点：数据根目录为当前目录下的 data/flickr8k ----------
    data_root = "./data/flickr8k"   # 或 "data/flickr8k"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = Flickr8kMini(data_root, processor, num_images=200)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    model = MiniBLIP2(device=device).to(device)
    
    # 只训练 Mini Q-Former 和 projection
    trainable_params = list(model.qformer.parameters()) + list(model.projection.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    
    epochs = 30
    loss_history = []
    best_loss = float('inf')
    
    # 创建输出目录
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./report", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            captions = batch['caption']
            
            tokenized = model.tokenizer(
                captions,
                padding='max_length',
                max_length=40,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            loss = model(pixel_values, input_ids, attention_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./outputs/mini_blip2_best.pt")
            print("Saved best model.")
    
    # 绘制 loss 曲线
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig("./report/loss_curve.png")
    plt.show()
    
if __name__ == "__main__":
    train()