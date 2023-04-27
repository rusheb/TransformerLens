from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components.layer_norm import LayerNorm
from transformer_lens.components.layer_norm_pre import LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import gelu_new, solu, gelu_fast


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


