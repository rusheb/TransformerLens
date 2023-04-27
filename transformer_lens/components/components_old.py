from typing import Union, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import logging
from jaxtyping import Float, Int

from functools import *

from transformer_lens.components.attention import Attention
from transformer_lens.components.layer_norm import LayerNorm
from transformer_lens.components.layer_norm_pre import LayerNormPre
from transformer_lens.components.rms_norm_pre import RMSNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import gelu_new, solu, gelu_fast
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.FactoredMatrix import FactoredMatrix

from fancy_einsum import einsum

from transformer_lens.past_key_value_caching import (
    HookedTransformerKeyValueCacheEntry,
)





class RMSNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, HookedTransformerConfig], length: Optional[int] = None
    ) -> object:

        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x = self.hook_normalized(x / scale)  # [batch, pos, length]
        return x * self.w


