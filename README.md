# 🧠 Vision Transformers Collection in PyTorch

This project implements three **state-of-the-art Vision Transformer (ViT)** variants in PyTorch:

- ✅ **ATS-ViT** (Adaptive Token Sampling)
- 🧩 **CaiT** (Class-Attention in Image Transformers)
- ⚡️ **CCT** (Compact Convolutional Transformer)

Each model is implemented in a modular and readable format, and can be easily extended or trained on your own datasets.

---

## 📚 Model Overview

| Model     | Highlights                                                                 | File         |
|-----------|---------------------------------------------------------------------------|--------------|
| **ATS-ViT** | Token sampling for efficiency; progressively reduces tokens per layer     | `ats_vit.py` |
| **CaiT**   | Separate class-attention blocks; deeper transformers with LayerScale       | `cait.py`    |
| **CCT**    | CNN tokenizer + transformer; lightweight for mobile and low-resource use   | `cct.py`     |

---

## 🔧 Installation

```bash
git clone https://github.com/your_username/vit-pytorch-models.git
cd vit-pytorch-models
pip install -r requirements.txt
```

---

## 🚀 Quick Start: ATS-ViT

```python
from ats_vit import ViT

model = ViT(
    image_size=224, patch_size=16, num_classes=1000,
    dim=768, depth=12,
    max_tokens_per_depth=(197, 128, 64, 32, 16, 8, 8, 8, 8, 8, 8, 8),
    heads=12, mlp_dim=3072,
    dropout=0.1, emb_dropout=0.1
)
```

---

## 🧩 CaiT

```python
from cait import CaiT
model = CaiT(
    image_size=224, patch_size=16, num_classes=1000,
    dim=768, depth=12, cls_depth=2,
    heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1
)
```

---

## ⚡️ CCT

```python
from cct import cct_7
model = cct_7(img_size=224, num_classes=1000)
```

---

## 🧪 Training
```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# training loop
model = ViT(...)  # or CaiT(...), cct_7(...)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for imgs, labels in trainloader:
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
---

## 📊 Evaluation
```python
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for imgs, labels in testloader:
        preds = model(imgs)
        correct += (preds.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    print(f"Accuracy: {correct / total * 100:.2f}%")
```

## 📈 Performance Benchmark (CIFAR-10 Example)

| Model   | Params | Top-1 Acc | Notes               |
| ------- | ------ | --------- | ------------------- |
| ATS-ViT | 85M    | 95.1%     | With token sampling |
| CaiT    | 86M    | 96.2%     | Deeper, stable      |
| CCT-7   | 21M    | 94.7%     | Light & efficient   |

## 🔗 References
An Image is Worth 16x16 Words (ViT)

CaiT: Going deeper with Image Transformers

Compact Convolutional Transformers

ATS-ViT 参考自 lucidrains/vit-pytorch

🧑‍💻 Author
Maintained by [Xiyuan Chen]

📫 Contact: xiyuan.chen23@qq.com

🔗 GitHub: @bichon frise

 
