from typing import Union, Dict

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class RMSNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """RMSNormPre - LayerNormPre without the centering and bias (RMS = Root Mean Square)"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
            self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        return self.hook_normalized(x / scale)  # [batch, pos, length]

