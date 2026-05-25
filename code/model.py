import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, OPTForCausalLM, AutoTokenizer

class MiniQFormer(nn.Module):
    """
    简化版 Q-Former: 包含可学习 queries，与图像特征进行 cross-attention
    输出形状: (batch, num_queries, dim)
    """
    def __init__(self, vision_dim=768, num_queries=32, num_heads=8, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, vision_dim))
        
        # Cross-attention + FFN 的简单堆叠
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(vision_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vision_dim, vision_dim * 4),
                nn.GELU(),
                nn.Linear(vision_dim * 4, vision_dim)
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(vision_dim) for _ in range(num_layers)])

    def forward(self, vision_features):
        """
        vision_features: (batch, seq_len, vision_dim) 来自 CLIP ViT 的 patch tokens
        """
        batch_size = vision_features.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)  # (b, num_q, dim)
        
        for attn, ff, ln in zip(self.cross_attn_layers, self.ffns, self.layer_norms):
            # Cross-attention: queries 作为 Q, vision_features 作为 K,V
            attn_out, _ = attn(queries, vision_features, vision_features)
            queries = queries + attn_out
            queries = ln(queries)
            # FFN
            ffn_out = ff(queries)
            queries = queries + ffn_out
        return queries  # (batch, num_queries, dim)


class MiniBLIP2(nn.Module):
    """
    完整模型：冻结 CLIP ViT + OPT，训练 MiniQFormer + Projection
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # 1. 冻结视觉编码器
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = self.vision_encoder.float()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # 2. Mini Q-Former
        self.qformer = MiniQFormer(vision_dim=768, num_queries=32)
        
        # 3. 投影层
        self.projection = nn.Linear(768, 768, bias=True)
        
        # 4. 冻结语言解码器 OPT-125M
        self.lm = OPTForCausalLM.from_pretrained("facebook/opt-125m")
        self.lm = self.lm.float()
        for param in self.lm.parameters():
            param.requires_grad = False
        
        # 获取 tokenizer 和 embedding 层
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm_embed = self.lm.get_input_embeddings()
        
        self.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        batch_size = pixel_values.size(0)
        
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values)
            vision_features = vision_outputs.last_hidden_state  # (b, 50, 768)
        
        query_outputs = self.qformer(vision_features)
        visual_tokens = self.projection(query_outputs)  # (b, num_q, 768)
        
        text_embeds = self.lm_embed(input_ids)  # (b, text_len, 768)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        
        full_attention_mask = torch.ones(batch_size, inputs_embeds.size(1), device=self.device)
        
        if labels is None:
            labels = input_ids
        full_labels = torch.full_like(inputs_embeds[:, :, 0], -100, dtype=torch.long)
        full_labels[:, visual_tokens.size(1):] = labels
        
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True
        )
        return outputs.loss
    
    @torch.no_grad()
    def generate(self, pixel_values, max_length=30):
        self.eval()
        batch_size = pixel_values.size(0)
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs.last_hidden_state
        query_outputs = self.qformer(vision_features)
        visual_tokens = self.projection(query_outputs)  # (b, num_q, 768)

        # 创建 BOS token ids (batch, 1)
        bos_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
        bos_embeds = self.lm_embed(bos_ids)  # (b, 1, 768)

        # 拼接视觉 token 和 BOS embedding
        inputs_embeds = torch.cat([visual_tokens, bos_embeds], dim=1)

        generated_ids = self.lm.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length + inputs_embeds.size(1),
            do_sample=False,
            num_beams=1,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # 解码生成的 ids，跳过特殊 token
        full_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return full_text