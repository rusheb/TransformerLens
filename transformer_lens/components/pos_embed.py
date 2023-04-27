from typing import Union, Dict

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg.n_ctx, self.cfg.d_model))

    def forward(
            self, tokens: Int[torch.Tensor, "batch pos"], past_kv_pos_offset: int = 0
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Tokens have shape [batch, pos]
        past_kv_pos_offset is the length of tokens in the past_kv_cache (if used, defaults to zero if unused)
        Output shape [pos, d_model] - will be broadcast along batch dim"""

        tokens_length = tokens.size(-1)
        pos_embed = self.W_pos[
                    past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
                    ]  # [pos, d_model]
        broadcast_pos_embed = einops.repeat(
            pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
        )  # [batch, pos, d_model]
        return broadcast_pos_embed.clone()

