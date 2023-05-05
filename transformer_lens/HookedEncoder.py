from typing import Dict

import torch
from jaxtyping import Float
from torch import nn

from transformer_lens import HookedTransformerConfig
from transformer_lens.components import BertEmbed, BertBlock
from transformer_lens.hook_points import HookedRootModule


class HookedEncoder(HookedRootModule):
    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        self.embed = BertEmbed(self.cfg)
        self.blocks = nn.ModuleList(
            [
                BertBlock(self.cfg) for _ in range(self.cfg.n_layers)
            ]
        )

    def forward(self, x: Float[torch.tensor, "batch pos"], token_type_ids=None):
        resid = self.embed(x, token_type_ids)

        for block in self.blocks:
            resid = block(resid)

        return resid
