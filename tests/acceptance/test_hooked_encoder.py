from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig

from transformer_lens.components import BertEmbed, MaskedAttention
from transformer_lens.HookedEncoderConfig import HookedEncoderConfig


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
        "d_head": hf_config.hidden_size,
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

# def test_bert_attention():
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#     sequence = "one token two tokens [MASK]"
#     encoding = tokenizer(sequence, return_tensors="pt")
#     input_ids = encoding["input_ids"]
#     cfg = HookedEncoderConfig(
#         d_vocab=28996,
#         d_model=768,
#         n_ctx=512,
#         n_heads=12,
#         d_head=768,
#         eps=1e-12,
#     )
#
#     hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
#     hf_embed = hf_bert.bert.embeddings
#     embed_out = hf_embed(input_ids)
#     our_attention = MaskedAttention(cfg)
#
#     state_dict = convert_bert_attention_weights(hf_bert, cfg)
#
#     our_attention.load_state_dict(state_dict)
#
#     hf_attention_self = hf_bert.bert.encoder.layer[0].attention.self
#     hf_attention_out = hf_bert.bert.encoder.layer[0].attention.output.dense
#     hf_attention_self_output = hf_attention_self(embed_out)[0]
#     their_attention_output = hf_attention_out(hf_attention_self_output)
#
#     # state_dict = convert_bert_attention_weights(
#     #     hf_attention_self, hf_attention_out
#     # )
#     # our_attention.load_state_dict(state_dict)
#     # breakpoint()
#
#     our_attention_output = our_attention(embed_out)
#
#     assert our_attention_output.shape == their_attention_output.shape
#
#
# def convert_bert_attention_weights(
#         bert, cfg: HookedEncoderConfig
# ):
#     state_dict = {}
#
#     for l in range(cfg.n_layers):
#         attention = bert.encoder.layer[l].attention
#         TODO this won't work because we are overwriting thee values every time into the statedict
#           and also we are not doing multi-headed attention

#         state_dict["W_Q"] = attention.self.query.weight
#         state_dict["b_Q"] = attention.self.query.bias
#
#         state_dict["W_K"] = attention.self.key.weight
#         state_dict["b_K"] = attention.self.key.bias
#
#         state_dict["W_V"] = attention.self.value.weight
#         state_dict["b_V"] = attention.self.value.bias
#
#         state_dict["W_O"] = attention.output.dense.weight
#         state_dict["b_O"] = attention.output.dense.bias
#
#     return state_dict
