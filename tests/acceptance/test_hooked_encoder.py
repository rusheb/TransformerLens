from einops import einops
from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig

from transformer_lens import HookedTransformerConfig
from transformer_lens.components import BertEmbed, Attention, BertBlock
from transformer_lens.HookedEncoderConfig import HookedEncoderConfig


def test_bert_embed_one_sentence():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer("Hello, world!", return_tensors="pt")
    input_ids = encoding["input_ids"]

    embed, huggingface_embed = load_embed()

    assert_close(embed(input_ids), huggingface_embed(input_ids))


def test_bert_embed_two_sentences():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer("First sentence.", "Second sentence.", return_tensors="pt")
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]
    embed, huggingface_embed = load_embed()

    assert_close(
        embed(input_ids, token_type_ids=token_type_ids),
        huggingface_embed(input_ids, token_type_ids=token_type_ids),
    )


def load_embed():
    cfg = convert_hf_model_cfg()
    embed = BertEmbed(cfg)

    huggingface_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    huggingface_embed = huggingface_bert.bert.embeddings

    state_dict = convert_bert_embedding_weights(huggingface_bert, cfg)
    embed.load_state_dict(state_dict)

    return embed, huggingface_embed


def convert_hf_model_cfg() -> HookedTransformerConfig:
    hf_config = AutoConfig.from_pretrained("bert-base-cased")
    cfg_dict = {
        "d_vocab": hf_config.vocab_size,
        "d_model": hf_config.hidden_size,
        "n_ctx": hf_config.max_position_embeddings,
        "n_heads": hf_config.num_attention_heads,
        "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
        "act_fn": "gelu",
        "n_layers": hf_config.num_hidden_layers,
        "eps": hf_config.layer_norm_eps,
        "attention_dir": "bidirectional",
    }
    return HookedTransformerConfig.from_dict(cfg_dict)


def convert_bert_embedding_weights(bert, cfg: HookedEncoderConfig):
    return {
        "word_embed.W_E": bert.bert.embeddings.word_embeddings.weight,
        "pos_embed.W_pos": bert.bert.embeddings.position_embeddings.weight,
        "token_type_embed.W_token_type": bert.bert.embeddings.token_type_embeddings.weight,
        "ln.w": bert.bert.embeddings.LayerNorm.weight,
        "ln.b": bert.bert.embeddings.LayerNorm.bias,
    }


def test_bert_attention():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    cfg = convert_hf_model_cfg()

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    hf_embed = hf_bert.bert.embeddings
    embed_out = hf_embed(input_ids)

    our_attention = Attention(cfg)
    hf_attention = hf_bert.bert.encoder.layer[0].attention

    attention = hf_bert.bert.encoder.layer[0].attention
    state_dict = {
        "W_Q": einops.rearrange(attention.self.query.weight, "(i h) m -> i m h", i=cfg.n_heads),
        "b_Q": einops.rearrange(attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads),
        "W_K": einops.rearrange(attention.self.key.weight, "(i h) m -> i m h", i=cfg.n_heads),
        "b_K": einops.rearrange(attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads),
        "W_V": einops.rearrange(attention.self.value.weight, "(i h) m -> i m h", i=cfg.n_heads),
        "b_V": einops.rearrange(attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads),
        "W_O": einops.rearrange(attention.output.dense.weight, "m (i h) -> i h m", i=cfg.n_heads),
        "b_O": attention.output.dense.bias
    }
    our_attention.load_state_dict(state_dict, strict=False)

    our_attention_out = our_attention(embed_out, embed_out, embed_out)
    hf_self_attention_out = hf_attention.self(embed_out)[0]
    hf_attention_out = hf_attention.output.dense(hf_self_attention_out)
    assert_close(our_attention_out, hf_attention_out)


def test_bert_block():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]

    cfg = convert_hf_model_cfg()

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    hf_embed = hf_bert.bert.embeddings
    embed_out = hf_embed(input_ids)

    our_block = BertBlock(cfg)
    hf_block = hf_bert.bert.encoder.layer[0]

    state_dict = {
        "attn.W_Q": einops.rearrange(hf_block.attention.self.query.weight, "(i h) m -> i m h", i=cfg.n_heads),
        "attn.b_Q": einops.rearrange(hf_block.attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads),
        "attn.W_K": einops.rearrange(hf_block.attention.self.key.weight, "(i h) m -> i m h", i=cfg.n_heads),
        "attn.b_K": einops.rearrange(hf_block.attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads),
        "attn.W_V": einops.rearrange(hf_block.attention.self.value.weight, "(i h) m -> i m h", i=cfg.n_heads),
        "attn.b_V": einops.rearrange(hf_block.attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads),
        "attn.W_O": einops.rearrange(hf_block.attention.output.dense.weight, "m (i h) -> i h m", i=cfg.n_heads),
        "attn.b_O": hf_block.attention.output.dense.bias,
        "ln1.w": hf_block.attention.output.LayerNorm.weight,
        "ln1.b": hf_block.attention.output.LayerNorm.bias,
        "mlp.W_in": einops.rearrange(hf_block.intermediate.dense.weight, "mlp model -> model mlp"),
        "mlp.b_in": hf_block.intermediate.dense.bias,
        "mlp.W_out": einops.rearrange(hf_block.output.dense.weight, "model mlp -> mlp model"),
        "mlp.b_out": hf_block.output.dense.bias,
        "ln2.w": hf_block.output.LayerNorm.weight,
        "ln2.b": hf_block.output.LayerNorm.bias
    }
    our_block.load_state_dict(state_dict, strict=False)

    our_block_out = our_block(embed_out)
    hf_block_out = hf_block(embed_out)
    assert_close(our_block_out, hf_block_out[0])
