# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成 Image Captioning。

输入：图片  
输出：英文 caption

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：前 200 张图片

## 4. 模型结构

请说明自己的 Mini-BLIP2 结构，例如：

```text
Image → Frozen Vision Encoder → Mini Q-Former → Projection Layer → Frozen Language Decoder → Caption
```

### 4.1 Vision Encoder

填写使用的视觉编码器，openai/clip-vit-base-patch32，输出 patch tokens (50, 768)。

### 4.2 Mini Q-Former

说明自己实现的 Mini Q-Former：

- query token 数量：32
- hidden size：768
- Transformer 层数：2 层交叉注意力 + FFN（未使用自注意力）
- 是否使用 cross-attention：否

### 4.3 Language Decoder

填写使用的语言解码器，facebook/opt-125m，自回归生成文本。

## 5. 训练设置

请填写：

- 训练数据量：200张图片
- epoch：30
- batch size：4
- learning rate：1e-4
- optimizer：AdamW（weight decay=0.01）
- loss function：文本 token 交叉熵损失（视觉 token 位置不参与计算）
- 冻结的模块：CLIP ViT（~86M 参数）、OPT-125M（~125M 参数）
- 训练的模块：Mini Q-Former（~2.4M 参数）、投影层（~0.59M 参数）

## 6. 训练过程

训练损失曲线如下图所示（loss_curve.png 在同一目录下）：

https://loss_curve.png

每轮平均损失变化如下（从训练日志提取）：

Epoch	Average Loss
1	3.9963
2	3.4527
3	3.3322
4	3.2488
5	3.1626
6	3.1018
7	3.0503
8	3.0074
9	2.9717
10	2.9453
11	2.9177
12	2.8964
13	2.8761
14	2.8569
15	2.8511
16	2.8314
17	2.8101
18	2.7956
19	2.7815
20	2.7690
21	2.7572
22	2.7439
23	2.7367
24	2.7273
25	2.7186
26	2.7044
27	2.6972
28	2.6846
29	2.6769
30	2.6715

## 7. 生成结果展示

随机选取的测试图片: ['1177994172_10d143cb8d.jpg', '1213336750_2269b51397.jpg', '1130369873_d80a1aa59c.jpg', '1119015538_e8e796281e.jpg']
Image: 1177994172_10d143cb8d.jpg
  True: Two blonde boys , one in a camouflage shirt and the other in blue , are having a water fight .
  Pred: Two boys are squirting water at each other .

Image: 1213336750_2269b51397.jpg
  True: A man in a black jacket is taking a photo of a man in a red jacket .
  Pred: A man takes a picture of another man in a red jacket .

Image: 1130369873_d80a1aa59c.jpg
  True: A brown dog is running through neck-deep water carrying a tennis ball .
  Pred: A brown dog is running through water in a tennis ball .

Image: 1119015538_e8e796281e.jpg
  True: A little tan dog with large ears running through the grass .
  Pred: A small dog is running through the grass .

## 8. 总结

请简要说明：

是否成功跑通训练：是。模型在 200 张小规模数据上成功训练并收敛。

生成效果如何：生成语句语法正确，能识别图片核心主体（如狗、女孩、动作等）。受限于数据量，部分细节丢失（如颜色、数量），多样性有限。验证了冻结双端、训练中间桥接模块的范式有效。

遇到了什么问题：小数据量易过拟合；Mini Q-Former 结构简单，生成描述有时重复（如 “a black and white dog and a black and white dog”）。

如果继续改进，可以怎么做：

扩充训练数据或使用数据增强；

增加 Mini Q-Former 层数，加入自注意力与对比学习损失；

使用 beam search 替代贪心搜索提升连贯性。

## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 录制工具：未使用特定录制工具（直接使用 DeepSeek 分享链接）
- 对话链接：https://chat.deepseek.com/share/wpmai4brpn0ywqe722
- 使用的 AI 模型：DeepSeek
- 累计对话时长 / 会话数：单轮对话，AI 思考用时 55 秒，回答内容较长

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：

```text
AI 帮助设计了完整的 Mini-BLIP2 复现流程，包括数据目录结构、Mini Q-Former 模型定义、训练脚本和生成脚本的完整代码实现。数据预处理和训练损失计算中的 labels 对齐等关键细节均由 AI 提供方案。数据下载和环境配置由自己独立完成，AI 建议的 batch size 和 num_queries 参数在实际运行中根据显存限制进行了调整。
```

## 10. Git 提交记录

请填写本次复现的代码仓库与提交历史（对应 requirements.md 第 9.2 节）。

- 仓库地址：https://github.com/melannana/blip2/tree/main
- 总 commit 数：8

粘贴 `git log --oneline` 输出（或截图）：

```text
b29ec75 Delete 陈致远 directory
4220de5 Add files via upload
edac08a Add empty data/ placeholder
b6011ba Translate README to Chinese and document data/ folder
54a0028 Add anti-cheat requirements: AI chat log and granular git commits
d6cb42d Add Mini-BLIP2 reproduction brief
bd9d0da Add dataset download notice to README
2444c94 Add basic project structure and docs
```
