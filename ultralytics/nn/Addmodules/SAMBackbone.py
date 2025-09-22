# sam_backbone.py
from segment_anything import sam_model_registry
import torch.nn as nn

class SAMBackbone(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        sam = sam_model_registry['vit_b'](checkpoint=ckpt)
        self.encoder = sam.image_encoder  # 只要 encoder

    def forward(self, x):
        with torch.no_grad():           # 如果不想更新SAM参数
            x = self.encoder(x)         # [B,C,H,W]
        return x