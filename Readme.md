# [Extracting Entity-Aspect-Opinion-Sentiment from Running Man Vietnam Comments using Transformer-based model]

[![Python 3.13](https://img.shields.io/badge/python-3.13-yellow)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20/%20Transformers-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Introduction
This project aims to build a deep learning model to extract Name Entity, Aspect, Opinion, and Sentiment from comments on the Running Man Vietnam.

## Model Architecture
This project implement Transformer-Encoder with Attetion

- **Embedding Layer:** Using PhoBERT để biểu diễn từ.
- **Encoder:** Using 2 layers - Transformer Encoder with 256 units.
- **Optimizer:** AdamW Optimizer với $\text{learning rate} = 2 \times 10^{-5}$.
- **Loss Function:** Weighted Cross-Entropy Loss with Label Smoothing technique.

## Data Pipeline
1. **Text Cleaning:** Loại bỏ nhiễu, chuẩn hóa Unicode (NFC/NFD).
2. **Tokenization:** Using Underthesea.
3. **Padding & Masking:** Đảm bảo độ dài đầu vào cố định $L=256$.

## Install & Using

### Environment
- Python 3.8+
- Model files: Transformer.pth, config.json
```bash
cd app
pip install -r requirements.txt
```