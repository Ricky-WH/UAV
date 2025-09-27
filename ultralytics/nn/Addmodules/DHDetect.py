import torch
import torch.nn as nn
import math
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Detect


class DHDetect(Detect):
    def __init__(self, nc=80, ch=(), use_fusion=True):
        """Initialize DHDetect with separate classification and regression branches."""
        super().__init__(nc, ch)
        self.use_fusion = use_fusion
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

        # Classification branch
        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )

        # Regression branch
        self.reg_convs = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )

        # Feature fusion if enabled
        if self.use_fusion:
            self.cls_attention = nn.ModuleList(
                nn.Sequential(
                    Conv(c3, c2, 1),
                    nn.Sigmoid()
                ) for _ in ch
            )
            self.reg_attention = nn.ModuleList(
                nn.Sequential(
                    Conv(c2, c3, 1),
                    nn.Sigmoid()
                ) for _ in ch
            )

    def forward(self, x):
        """
        Forward pass of the decoupled head.
        
        Args:
            x (List[torch.Tensor]): List of feature maps from backbone
            
        Returns:
            List[torch.Tensor]: List of detection outputs
        """
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            if self.use_fusion:
                # Feature fusion using attention mechanism
                cls_feat = self.cls_convs[i][:-1](x[i])
                reg_feat = self.reg_convs[i][:-1](x[i])
                
                cls_attn = self.cls_attention[i](cls_feat)
                reg_attn = self.reg_attention[i](reg_feat)
                
                cls_out = self.cls_convs[i][-1](cls_feat * reg_attn)
                reg_out = self.reg_convs[i][-1](reg_feat * cls_attn)
            else:
                # Direct separate prediction without fusion
                cls_out = self.cls_convs[i](x[i])
                reg_out = self.reg_convs[i](x[i])
            
            # Combine outputs
            x[i] = torch.cat((reg_out, cls_out), 1)

        if self.training:
            return x

        # Apply DFL and create final predictions
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases for better training stability."""
        for m in self.cls_convs:
            final_layer = m[-1]
            final_layer.bias.data[:] = math.log(5 / self.nc / (640 / 32) ** 2)
        for m in self.reg_convs:
            final_layer = m[-1]
            final_layer.bias.data[:] = 1.0