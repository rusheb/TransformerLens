#%%
from easy_transformer import EasyTransformer
import logging
import sys
from ioi_circuit_extraction import *
import optuna
from ioi_dataset import *
import IPython
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
)  # helper functions
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import (
    EasyTransformer,
    TransformerBlock,
    MLP,
    Attention,
    LayerNormPre,
    PosEmbed,
    Unembed,
    Embed,
)
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
    get_act_hook,
)
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Iterable
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import spacy
import re
from einops import rearrange
import einops
from pprint import pprint
import gc
from datasets import load_dataset
from IPython import get_ipython
import matplotlib.pyplot as plt
import random as rd

from ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    ABBA_TEMPLATES,
)
from ioi_utils import (
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)
from easy_transformer.utils import print_gpu_mem

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

def e():
    torch.cuda.empty_cache()
# %%
model = EasyTransformer("gpt2", use_attn_result=True).cuda()
N = 100
# ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
# abca_dataset = ioi_dataset.gen_flipped_prompts("S2")  # we flip the second b for a random c
MAX_CONTEXT = 1024
#%% [markdown] Get 10K OWT samples. Seems already randomized
owt_train_text = load_dataset("stas/openwebtext-10k")["train"]["text"]
owt_train_text = owt_train_text[:20]
owt_train_toks = torch.zeros(size=(len(owt_train_text), MAX_CONTEXT)).long()
owt_train_toks += model.tokenizer.pad_token_id
toks_raw = model.tokenizer(owt_train_text)["input_ids"]
toks = torch.zeros(size=(len(owt_train_text), MAX_CONTEXT)).long()
toks += model.tokenizer.pad_token_id

for i in range(len(toks)):
    cur_tensor = torch.tensor(toks_raw[i][:MAX_CONTEXT])
    toks[i, : len(cur_tensor)] = cur_tensor
e()
#%%
all_log_probs = torch.zeros(size=(len(owt_train_text), MAX_CONTEXT))

for i in tqdm(range(len(toks_raw))):
    e()
    cur_tensor = torch.tensor(toks_raw[i][:MAX_CONTEXT]).cuda().unsqueeze(0)
    logits = model(cur_tensor.cuda())
    log_probs = t.nn.functional.log_softmax(logits, dim=-1)[0]
    all_log_probs[i][1:len(toks_raw[i])] = log_probs.detach().cpu()[torch.arange(0, min(1024, len(toks_raw[i])) - 1), cur_tensor[0][1:]] 

    if i == 0:
        assert 0.9 < torch.exp(all_log_probs[0][8]) < 1.0, "You didn't complete Adolf -> Hitler"

    # some sanity checking of what the top predictions are, probably keep it's finnicky
    # for i in range(20):
    #     print(i, model.tokenizer.decode(cur_tensor[0][i:i+1]))
    # mm = torch.max(log_probs, dim=1)
    # for i in range(20):
    #     print(i, model.tokenizer.decode(mm[1][i:i+1]))
#%%

BATCH_SIZE = 10

def get_tokens(batch):
    """Turn strings into token ids"""
    list_of_toks = model.tokenizer(batch, padding=True)["input_ids"]
    if len(list_of_toks[0]) < MAX_CONTEXT:
        warnings.warn("Adding padding to batch")
        for i in range(len(list_of_toks)):
            list_of_toks[i] += [model.tokenizer.pad_token_id] * (MAX_CONTEXT - len(list_of_toks[i]))
    return torch.Tensor(list_of_toks)[:,:MAX_CONTEXT].long()

def log_probs_correct(model, dataset):
    assert len(dataset) % BATCH_SIZE == 0, (len(dataset), BATCH_SIZE)
    all_log_probs = torch.zeros(size=(len(dataset), MAX_CONTEXT))
    
    for i in range(len(dataset) // BATCH_SIZE): 
        logits = model(get_tokens(dataset[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]).cuda())
        log_probs = t.nn.functional.log_softmax(logits, dim=-1)[0].detach().clone()
        all_log_probs[i * BATCH_SIZE : (i + 1)*BATCH_SIZE][1:] = log_probs[torch.arange(0, min(1024, len(dataset[i])) - 1), cur_tensor[0][1:]] 

log_probs_correct(model, owt_train_text)

    metric = ExperimentMetric(metric=log_probs_correct, dataset=toks, relative_metric = True)
    config = AblationConfig(abl_type="mean", target_module="attn_head", head_circuit="z",  cache_means=True, verbose=True)
    abl = EasyAblation(model, config, metric)
    result = abl.run_ablation()

# #%%
# for i, text in enumerate(owt_train_text):
#     toks = torch.Tensor(model.tokenizer(owt_train_text[3:5])["input_ids"])[:MAX_CONTEXT].long().cuda().unsqueeze(0)
#     owt_train_toks[i, :toks.shape[1]] = toks
# #%%
# for data in owt_train_text:
#     toks = torch.Tensor(model.tokenizer(data)["input_ids"])[:MAX_CONTEXT].long().cuda().unsqueeze(0)
#     model.reset_hooks()
#     logits = model(toks)
#     log_probs = t.nn.functional.log_softmax(logits, dim=-1).clone()

#     metric = ExperimentMetric()    
# %%
