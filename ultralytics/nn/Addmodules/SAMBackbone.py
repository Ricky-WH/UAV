from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMBackbone(nn.Module):
    def __init__(self, in_channels=None, out_channels=512, ckpt=None):
        super().__init__()
        self.out_channels = out_channels
        self.ckpt = ckpt

        # 延迟初始化 SAM 模型与投影头，首次 forward 根据实际通道创建
        self.encoder = None
        self.hidden_size = None
        self.proj = None

    def _build_encoder(self):
        try:
            from transformers import Sam2Config, Sam2Model
        except Exception as e:
            raise ImportError("transformers is required for SAMBackbone. Please install it via 'pip install transformers'.") from e

        import os
        sam2 = None
        ckpt = self.ckpt
        if ckpt:
            try:
                if os.path.isdir(ckpt) or (os.path.isfile(ckpt) and os.path.isfile(os.path.join(os.path.dirname(ckpt), "config.json"))):
                    sam2 = Sam2Model.from_pretrained(ckpt, local_files_only=True)
            except Exception:
                sam2 = None

        if sam2 is None:
            config = None
            state_dict = None
            if ckpt and os.path.isfile(ckpt):
                try:
                    loaded = torch.load(ckpt, map_location="cpu")
                except Exception:
                    loaded = None
                if isinstance(loaded, dict):
                    cfg_dict = None
                    for k in ("config", "cfg", "model_cfg", "model_args"):
                        if k in loaded and isinstance(loaded[k], dict):
                            cfg_dict = loaded[k]
                            break
                    if cfg_dict is not None:
                        try:
                            from transformers import Sam2Config as _Sam2Config
                            config = _Sam2Config.from_dict(cfg_dict)
                        except Exception:
                            config = None
                    if any(isinstance(k, str) and (k.startswith("vision_encoder") or k.startswith("shared_image_embedding")) for k in loaded.keys()):
                        state_dict = loaded
                    elif "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                        sd = loaded["state_dict"]
                        if any(isinstance(k, str) and (k.startswith("vision_encoder") or k.startswith("shared_image_embedding")) for k in sd.keys()):
                            state_dict = sd

            if config is None:
                config = Sam2Config(
                    image_size=1024,
                    patch_size=16,
                    hidden_size=1280,
                    num_hidden_layers=40,
                    num_attention_heads=16,
                    intermediate_size=5120,
                    dropout_rate=0.0,
                    encoder_type="hiera",
                )
            sam2 = Sam2Model(config)
            if state_dict is not None:
                sam2.load_state_dict(state_dict, strict=False)

        self.encoder = getattr(sam2, "vision_encoder", None) or nn.Identity()
        self.hidden_size = getattr(sam2.config, "hidden_size", None)

    def forward(self, x: torch.Tensor):
        if self.encoder is None:
            self._build_encoder()

        enc_out = self.encoder(x)

        # 统一为 (B, C, H, W)
        if hasattr(enc_out, "last_hidden_state") and enc_out.last_hidden_state is not None:
            fx = enc_out.last_hidden_state
        elif hasattr(enc_out, "image_embeddings") and enc_out.image_embeddings is not None:
            fx = enc_out.image_embeddings
        elif isinstance(enc_out, dict):
            fx = None
            for v in enc_out.values():
                if torch.is_tensor(v):
                    fx = v
                    break
            fx = fx if fx is not None else enc_out.get("0", None)
        elif isinstance(enc_out, (list, tuple)) and len(enc_out) > 0:
            fx = enc_out[0]
        else:
            fx = enc_out

        if fx.ndim == 3:  # (B, L, C)
            b, l, c = fx.shape
            g = int(l ** 0.5)
            if g * g != l:
                g = int(l ** 0.5)
                gh = g
                gw = max(1, l // max(1, gh))
                fx = fx.transpose(1, 2).contiguous().view(b, c, gh, gw)
            else:
                fx = fx.transpose(1, 2).contiguous().view(b, c, g, g)
        elif fx.ndim == 4 and self.hidden_size is not None and fx.shape[1] != self.hidden_size and fx.shape[-1] == self.hidden_size:
            fx = fx.permute(0, 3, 1, 2).contiguous()

        if self.proj is None:
            in_ch = fx.shape[1]
            self.hidden_size = in_ch
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(),
            ).to(fx.device, dtype=fx.dtype)

        return self.proj(fx)