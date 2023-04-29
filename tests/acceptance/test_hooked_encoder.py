from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM

from transformer_lens.components import BertEmbed
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
    cfg = HookedEncoderConfig(
        d_vocab=28996,
        d_model=768,
        n_ctx=512,
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
