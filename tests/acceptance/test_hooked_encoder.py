from einops import einops
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


def test_bert_attention():
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
        "W_Q": einops.rearrange(hf_attention.self.query.weight, "(head_index d_head) d_model -> head_index d_model d_head", head_index=cfg.n_heads),
        "b_Q": einops.rearrange(hf_attention.self.query.bias, "(head_index d_head) -> head_index d_head", head_index=cfg.n_heads),
        "key.weight": hf_attention.self.key.weight,
        "key.bias": hf_attention.self.key.bias,
        "value.weight": hf_attention.self.value.weight,
        "value.bias": hf_attention.self.value.bias,
        "W_O": einops.rearrange(hf_attention.output.dense.weight, "d_model (head_index d_head) -> head_index d_head d_model", head_index=cfg.n_heads),
        "b_O": hf_attention.output.dense.bias
    }

    our_attention.load_state_dict(state_dict)

    our_attention_out = our_attention(embed_out)
    hf_self_attention_out = hf_attention.self(embed_out)[0]
    hf_attention_out = hf_attention.output.dense(hf_self_attention_out)
    assert_close(our_attention_out, hf_attention_out)


def convert_bert_attention_weights(
        bert, cfg: HookedEncoderConfig
):
    pass