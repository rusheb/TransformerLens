from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM

from transformer_lens.components import BertEmbed, MaskedAttention
from transformer_lens.HookedEncoderConfig import HookedEncoderConfig


def test_bert_attention():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequence = "one token two tokens [MASK]"
    encoding = tokenizer(sequence, return_tensors="pt")
    input_ids = encoding["input_ids"]
    cfg = HookedEncoderConfig(
        d_vocab=28996,
        d_model=768,
        n_ctx=512,
        n_heads=12,
        d_head=768,
        eps=1e-12,
    )

    hf_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    hf_embed = hf_bert.bert.embeddings
    embed_out = hf_embed(input_ids)
    our_attention = MaskedAttention(cfg)

    hf_attention_self = hf_bert.bert.encoder.layer[0].attention.self
    hf_attention_out = hf_bert.bert.encoder.layer[0].attention.output.dense

    state_dict = convert_hf_bert_attention_to_tl_masked_attention(
        hf_attention_self, hf_attention_out
    )
    our_attention.load_state_dict(state_dict)
    breakpoint()

    our_attention_output = our_attention(embed_out)
    hf_attention_self_output = hf_attention_self(embed_out)[0]
    their_attention_output = hf_attention_out(hf_attention_self_output)

    assert our_attention_output.shape == their_attention_output.shape


def convert_hf_bert_attention_to_tl_masked_attention(
    hf_attention_self, hf_attention_out
):
    tl_masked_attention_weights = {}
    hf_to_tl_name = {
        "query.weight": "W_Q",
        "key.weight": "W_K",
        "value.weight": "W_V",
        "query.bias": "b_Q",
        "key.bias": "b_K",
        "value.bias": "b_V",
        "weight": "W_O",
        "bias": "b_O",
    }
    for hf_param_name, param in hf_attention_self.named_parameters():
        tl_param_name = hf_to_tl_name[hf_param_name]
        tl_masked_attention_weights[tl_param_name] = param
    for hf_param_name, param in hf_attention_out.named_parameters():
        tl_param_name = hf_to_tl_name[hf_param_name]
        tl_masked_attention_weights[tl_param_name] = param
    return tl_masked_attention_weights


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
    cfg = HookedEncoderConfig(
        d_vocab=28996,
        d_model=768,
        n_ctx=512,
        n_heads=12,
        d_head=768,
        eps=1e-12,
    )

    embed = BertEmbed(cfg)
    my_parameters = list(embed.named_parameters())

    huggingface_bert = BertForMaskedLM.from_pretrained("bert-base-cased")
    huggingface_embed = huggingface_bert.bert.embeddings
    their_parameters = list(huggingface_embed.named_parameters())

    state_dict = {
        my_name: their_param
        for (my_name, _), (_, their_param) in zip(my_parameters, their_parameters)
    }

    embed.load_state_dict(state_dict)

    return embed, huggingface_embed
