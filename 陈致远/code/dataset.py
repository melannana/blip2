import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class Flickr8kMini(Dataset):
    def __init__(self, data_root, processor, split='train', num_images=200):
        """
        data_root: 包含 Images/ 和 captions.txt 的目录
        processor: CLIPProcessor 用于图片预处理
        split: 本简化版不分train/val，全部用于训练
        num_images: 使用前 num_images 张图片
        """
        self.data_root = data_root
        self.processor = processor
        captions_path = os.path.join(data_root, 'captions.txt')
        df = pd.read_csv(captions_path)
        
        # 获取前 num_images 个唯一图片名
        unique_images = df['image'].unique()[:num_images]
        self.samples = []
        for img_name in unique_images:
            img_captions = df[df['image'] == img_name]['caption'].tolist()
            for cap in img_captions:
                self.samples.append((img_name, cap))
        
        print(f"Loaded {len(self.samples)} (image, caption) pairs from {len(unique_images)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.data_root, 'Images', img_name)
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return {
            'pixel_values': pixel_values,
            'caption': caption,
            'img_name': img_name
        }