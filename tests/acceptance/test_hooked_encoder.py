from einops import einops
import torch
from fancy_einsum import einsum
from torch import nn
from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig

from transformer_lens.components import BertEmbed, MaskedAttention
from transformer_lens.HookedEncoderConfig import HookedEncoderConfig
from transformer_lens.utils import get_corner


def test_bert_embed_one_sentence():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer("Hello, world!", return_tensors="pt")
    input_ids = encoding["input_ids"]

    embed, huggingface_embed = load_weights_from_huggingface()

    assert_close(embed(input_ids), huggingface_embed(input_ids))


def test_bert_embed_two_sentences():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer("First sentence.", "Second sentence.", return_tensors="pt")
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]
    embed, huggingface_embed = load_weights_from_huggingface()

    assert_close(
        embed(input_ids, token_type_ids=token_type_ids),
        huggingface_embed(input_ids, token_type_ids=token_type_ids),
    )


def load_weights_from_huggingface():
    cfg = convert_hf_model_cfg()
    embed = BertEmbed(cfg)

    huggingface_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    huggingface_embed = huggingface_bert.bert.embeddings

    state_dict = convert_bert_embedding_weights(huggingface_bert, cfg)
    embed.load_state_dict(state_dict)

    return embed, huggingface_embed


def convert_hf_model_cfg() -> HookedEncoderConfig:
    hf_config = AutoConfig.from_pretrained("bert-base-cased")
    cfg_dict = {
        "d_vocab": hf_config.vocab_size,
        "d_model": hf_config.hidden_size,
        "n_ctx": hf_config.max_position_embeddings,
        "n_heads": hf_config.num_attention_heads,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "n_layers": hf_config.num_hidden_layers,
        "eps": hf_config.layer_norm_eps,
    }
    return HookedEncoderConfig.from_dict(cfg_dict)


def convert_bert_embedding_weights(bert, cfg: HookedEncoderConfig):
    state_dict = {
        "word_embed.W_E": bert.bert.embeddings.word_embeddings.weight,
        "pos_embed.W_pos": bert.bert.embeddings.position_embeddings.weight,
        "token_type_embed.W_token_type": bert.bert.embeddings.token_type_embeddings.weight,
        "ln.w": bert.bert.embeddings.LayerNorm.weight,
        "ln.b": bert.bert.embeddings.LayerNorm.bias,
    }

    return state_dict


def test_bert_attention_load():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # TODO: change this
    sequence = "one token two tokens [MASK]"
    input_ids = tokenizer(sequence, return_tensors="pt")["input_ids"]

    cfg = convert_hf_model_cfg()

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    hf_embed = hf_bert.bert.embeddings
    embed_out = hf_embed(input_ids)

    # state_dict = convert_bert_attention_weights(hf_bert, cfg)
    our_attention = MaskedAttention(cfg)


    hf_attention = hf_bert.bert.encoder.layer[0].attention

    state_dict = {
        "query.weight": hf_attention.self.query.weight,
        "query.bias": hf_attention.self.query.bias,
        "key.weight": hf_attention.self.key.weight,
        "key.bias": hf_attention.self.key.bias,
        "value.weight": hf_attention.self.value.weight,
        "value.bias": hf_attention.self.value.bias,
    }

    our_attention.load_state_dict(state_dict)

    our_attention_out = our_attention(embed_out)
    hf_attention_out = hf_bert.bert.encoder.layer[0].attention.self(embed_out)[0]
    assert_close(our_attention_out, hf_attention_out)



def convert_bert_attention_weights(
        bert, cfg: HookedEncoderConfig
):
    state_dict = {}

    attention = bert.bert.encoder.layer[0].attention

    W_Q = einops.rearrange(attention.self.query.weight, "m (i h) -> i m h", i=cfg.n_heads)
    b_Q = einops.rearrange(attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads)

    W_K = einops.rearrange(attention.self.key.weight, "m (i h) -> i m h", i=cfg.n_heads)
    b_K = einops.rearrange(attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads)

    W_V = einops.rearrange(attention.self.value.weight, "m (i h) -> i m h", i=cfg.n_heads)
    b_V = einops.rearrange(attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads)

    W_O = einops.rearrange(attention.output.dense.weight, "(i h) m->i h m", i=cfg.n_heads)

    state_dict["W_Q"] = W_Q
    state_dict["b_Q"] = b_Q

    state_dict["W_K"] = W_K
    state_dict["b_K"] = b_K

    state_dict["W_V"] = W_V
    state_dict["b_V"] = b_V

    # TODO is output correct?

    state_dict["W_O"] = W_O
    state_dict["b_O"] = attention.output.dense.bias

    return state_dict


    # our_attention = MaskedAttention(cfg)
    #
    # state_dict = convert_bert_attention_weights(hf_bert, cfg)
    #
    # our_attention.load_state_dict(state_dict)
    #
    # hf_attention_self = hf_bert.bert.encoder.layer[0].attention.self
    # hf_attention_out = hf_bert.bert.encoder.layer[0].attention.output.dense
    # hf_attention_self_output = hf_attention_self(embed_out)[0]
    their_attention_output = hf_attention_out(hf_attention_self_output)

    # state_dict = convert_bert_attention_weights(
    #     hf_attention_self, hf_attention_out
    # )
    # our_attention.load_state_dict(state_dict)
    # breakpoint()
#
#     our_attention_output = our_attention(embed_out)
#
#     assert our_attention_output.shape == their_attention_output.shape
#
#
# def test_bert_attention_wip():
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#     sequence = "Hello, world!"
#
#     input_ids = tokenizer(sequence, return_tensors="pt")["input_ids"]
#
#     cfg = convert_hf_model_cfg()
#
#     hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
#     hf_embed = hf_bert.bert.embeddings
#     embed_out = hf_embed(input_ids)
#
#     hf_attention = hf_bert.bert.encoder.layer[0].attention
#
#     state_dict = {
#         "W_Q":  einops.rearrange(
#             hf_attention.self.query.weight, "m (i h) -> i m h", i=cfg.n_heads
#         ),
#         "b_Q": einops.rearrange(
#             hf_attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads
#         )
#     }
#
#     q = JustQ(cfg)
#     q.load_state_dict(state_dict)
#
#     hf_q = hf_attention.self.query
#
#     q_out = q(embed_out)
#     hf_q_out = hf_q(embed_out)
#     hf_q_reshape = einops.rearrange(hf_q_out, "batch pos (head_index d_head) -> batch pos head_index d_head", head_index=cfg.n_heads)
#
#     assert_close(q(embed_out), hf_q_reshape)
#
#
# class JustQ(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
#         self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
#
#     def forward(self, resid):
#         return einsum(
#             "batch pos d_model, head_index d_model d_head \
#             -> batch pos head_index d_head",
#             resid,
#             self.W_Q,
#         ) + self.b_Q
#
#
