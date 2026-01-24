# StainNet

Scaling Self-Supervised Foundation Models on **Immunohistochemistry (IHC)** and **Special Stains** for Computational Pathology.

[[ArXiv]](https://arxiv.org/abs/2512.10326) [[StainNet-Base]](https://huggingface.co/JWonderLand/StainNet-Base) [[StainNet-Small]](https://huggingface.co/JWonderLand/StainNet)

StainNet is a collection of self-supervised foundation models specifically designed for IHC and special stains in pathology images.

## Introduction

The StainNet models are pre-trained using the DINO [1] self-supervised learning method on over 1.4 million patches extracted from 20,231 special staining whole slide images (WSIs) from the [HISTAI](https://arxiv.org/abs/2505.12120) [2] dataset.

We provide two versions of the model:

*   **[StainNet-Base](https://huggingface.co/JWonderLand/StainNet-Base)**: Based on the Vision Transformer Base/16 (ViT-Base/16) architecture.
*   **[StainNet-Small](https://huggingface.co/JWonderLand/StainNet)**: Based on the Vision Transformer Small/16 (ViT-Small/16) architecture, which is more lightweight.

## Quick Start

You can easily use the models for feature extraction via the `timm` library:

```python
import timm
import torch
import torchvision.transforms as transforms

# Load the pretrained model (StainNet-Base)
model = timm.create_model('hf_hub:JWonderLand/StainNet-Base', pretrained=True)

# Load the pretrained model (StainNet-Small)
# model = timm.create_model('hf_hub:JWonderLand/StainNet', pretrained=True)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

model = model.to('cuda')
model.eval()

# Dummy input (Batch size=1, Channels=3, H=224, W=224)
input_tensor = torch.randn([1, 3, 224, 224]).cuda()

with torch.no_grad():
    output = model(input_tensor)
    # StainNet-Base output dimension: [1, 768]
    # StainNet-Small output dimension: [1, 384]
    print(f"Output shape: {output.shape}")
```

## Citation

If StainNet is helpful to your research, please cite our work:

```bibtex
@misc{li2025stainnet,
      title={StainNet: A Special Staining Self-Supervised Vision Transformer for Computational Pathology}, 
      author={Jiawen Li and Jiali Hu and Xitong Ling and Yongqiang Lv and Yuxuan Chen and Yizhi Wang and Tian Guan and Yifei Liu and Yonghong He},
      year={2025},
      eprint={2512.10326},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.10326}, 
}
```

## References

1. [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
2. [HISTAI: An Open-Source, Large-Scale Whole Slide Image Dataset for Computational Pathology](https://arxiv.org/abs/2505.12120)
