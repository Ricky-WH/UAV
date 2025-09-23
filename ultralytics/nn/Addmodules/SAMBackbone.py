from transformers import Sam2Config, Sam2Model
import torch.nn as nn
import torch

class SAMBackbone(nn.Module):
    def __init__(self, _in_channels=None, ckpt="/root/Anti-UAV/sam2.1_hiera_large.pt"):
        super().__init__()

        # 手动创建模型配置
        config = Sam2Config(
            image_size=1024,
            patch_size=16,
            hidden_size=1280,
            num_hidden_layers=40,
            num_attention_heads=16,
            intermediate_size=5120,
            dropout_rate=0.0,
            encoder_type="hiera"
        )

        # 初始化模型结构
        self.sam2 = Sam2Model(config)

        # 加载本地 checkpoint（兼容多种保存格式）；若不匹配则跳过
        try:
            loaded_obj = torch.load(ckpt, map_location="cpu") if ckpt is not None else None
        except Exception as e:
            loaded_obj = None

        state_dict = None
        if isinstance(loaded_obj, dict):
            # 直接是权重字典
            if any(isinstance(k, str) and (k.startswith("vision_encoder") or k.startswith("shared_image_embedding")) for k in loaded_obj.keys()):
                state_dict = loaded_obj
            # 常见嵌套 state_dict
            elif "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
                sd = loaded_obj["state_dict"]
                if any(isinstance(k, str) and (k.startswith("vision_encoder") or k.startswith("shared_image_embedding")) for k in sd.keys()):
                    state_dict = sd

        if state_dict is not None:
            self.sam2.load_state_dict(state_dict, strict=False)
        # 若无法识别或不匹配，则使用随机初始化继续

        # 只取 vision_encoder（Sam2Model 使用 vision_encoder 命名）
        self.encoder = getattr(self.sam2, "vision_encoder", None) or nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            enc_out = self.encoder(x)

        # 将 Sam2VisionEncoderOutput/字典/元组 转为张量
        if hasattr(enc_out, "last_hidden_state") and enc_out.last_hidden_state is not None:
            x = enc_out.last_hidden_state
        elif hasattr(enc_out, "image_embeddings") and enc_out.image_embeddings is not None:
            x = enc_out.image_embeddings
        elif isinstance(enc_out, dict):
            # 取第一个张量条目
            for v in enc_out.values():
                if torch.is_tensor(v):
                    x = v
                    break
            else:
                x = enc_out.get("0", x)
        elif isinstance(enc_out, (list, tuple)) and len(enc_out) > 0:
            x = enc_out[0]
        else:
            x = enc_out

        return x