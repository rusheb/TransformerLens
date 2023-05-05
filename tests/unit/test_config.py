"""
Tests that verify than an arbitrary component (e.g. Embed) can be initialized using dict and object versions of HookedTransformerConfig and HookedEncoderConfig.
"""

import pytest
from transformer_lens.HookedEncoderConfig import HookedEncoderConfig
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components import Embed


def test_hooked_transformer_config_object():
    hooked_transformer_config = HookedTransformerConfig(
        n_layers=2, d_vocab=100, d_model=6, n_ctx=5, d_head=2, attn_only=True
    )
    Embed(hooked_transformer_config)


def test_hooked_transformer_config_dict():
    hooked_transformer_config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
        "attn_only": True,
    }
    Embed(hooked_transformer_config_dict)