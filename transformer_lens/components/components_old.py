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



# MLP Layers
class MLP(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in)
            + self.b_in
        )  # [batch, pos, d_mlp]
        if not self.cfg.act_fn.endswith("_ln"):
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return (
            einsum(
                "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )

# TODO
# not sure whether to fold this into MLP or not
class GatedMLP(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.W_gate = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model))

        # hook on gate output but before act_fn 
        self.hook_pre = HookPoint() # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in 
        self.hook_post = HookPoint() # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_gate))  # [batch, pos, d_mlp]
        if not self.cfg.act_fn.endswith("_ln"):
            post_act = self.hook_post(self.act_fn(pre_act) * einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in)
            + self.b_in)  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return (
            einsum(
                "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], block_index):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln1 = LayerNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(cfg)
        elif self.cfg.normalization_type == "RMS": 
            self.ln1 = RMSNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNorm(cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln1 = RMSNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNormPre(cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
            if not self.cfg.attn_only:
                self.ln2 = nn.Identity()
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )

        if not self.cfg.use_local_attn:
            self.attn = Attention(cfg, "global", block_index)
        else:
            assert self.cfg.attn_types is not None
            attn_type = self.cfg.attn_types[block_index]
            self.attn = Attention(cfg, attn_type, block_index)
        if not self.cfg.attn_only:
            if self.cfg.gated_mlp:
                self.mlp = GatedMLP(cfg)
            else: 
                self.mlp = MLP(cfg)

        self.hook_q_input = HookPoint()  # [batch, pos, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """A single Transformer block.

        Args:
            resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
            cache (HookedTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
            shortformer_pos_embed (torch.Tensor, optional): Only used for positional_embeddings_type == "shortformer". The positional embeddings. See HookedTransformerConfig for details. Defaults to None.

        Returns:
            _type_: _description_
        """
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]

        query_input = resid_pre
        key_input = resid_pre
        value_input = resid_pre
        
        if self.cfg.use_split_qkv_input:
            def add_head_dimension(tensor):
                return einops.repeat(tensor, "batch pos d_model -> batch pos n_heads d_model", n_heads=self.cfg.n_heads).clone()

            query_input = self.hook_q_input(add_head_dimension(query_input))
            key_input = self.hook_k_input(add_head_dimension(key_input))
            value_input = self.hook_v_input(add_head_dimension(value_input))

            if shortformer_pos_embed is not None:
                shortformer_pos_embed = add_head_dimension(shortformer_pos_embed)

        attn_out = self.hook_attn_out(
            # hook the residual stream states that are used to calculate the 
            # queries, keys and values, independently. 
            # Then take the layer norm of these inputs, and pass these to the attention module.
            self.attn(
                query_input = self.ln1(query_input) + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                key_input = self.ln1(key_input) + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                value_input = self.ln1(value_input),
                past_kv_cache_entry=past_kv_cache_entry,
            )
        )  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            resid_mid = self.hook_resid_mid(
                resid_pre + attn_out
            )  # [batch, pos, d_model]
            normalized_resid_mid = self.ln2(resid_mid)
            mlp_out = self.hook_mlp_out(
                self.mlp(normalized_resid_mid)
            )  # [batch, pos, d_model]
            resid_post = self.hook_resid_post(
                resid_mid + mlp_out
            )  # [batch, pos, d_model]
        elif self.cfg.parallel_attn_mlp:
            # Dumb thing done by GPT-J, both MLP and Attn read from resid_pre and write to resid_post, no resid_mid used.
            # In GPT-J, LN1 and LN2 are tied, in GPT-NeoX they aren't.
            normalized_resid_pre_2 = self.ln2(resid_pre)
            mlp_out = self.hook_mlp_out(
                self.mlp(normalized_resid_pre_2)
            )  # [batch, pos, d_model]
            resid_post = self.hook_resid_post(
                resid_pre + attn_out + mlp_out
            )  # [batch, pos, d_model]
        else:
            resid_post = self.hook_resid_post(
                resid_pre + attn_out
            )  # [batch, pos, d_model]
        return resid_post
