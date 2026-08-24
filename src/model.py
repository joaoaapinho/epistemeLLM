"""
Model loader.
"""

import os
import random

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from episteme import config

QUANT = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def set_seed(seed=config.SEED):
    """Define global seed, to be called on all entry points."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) #for hash ordering on data shuffle
    return seed


def load_tokenizer(model_id):
    """Load tokenizer - no GPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"

    assert tokenizer.pad_token is not None, f"{model_id} has no pad token."
    return tokenizer


def load_model(model_id, adapter_path=None, four_bit=True):
    """Return (model, tokenizer), model in eval mode on cuda:0.
    
    model_id: HF repo id or local path
    adapter_path: optional LoRA to stack on 4-bit base
    four_bit: False only for CPU debugging/ better GPU
    """

    tokenizer = load_tokenizer(model_id)

    kwargs = dict(
        device_map = {"": 0},
        attn_implementation = "sdpa", #no flash-attn
        dtype = torch.bfloat16,
    )

    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(**QUANT)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter_path is not None:
        # Lazy import so eval-only runs dont depend on peft
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        # Check adapter is attached
        assert any("lora" in n.lower() for n, _ in model.named_parameters()), (f"No LoRA params found after loading {adapter_path}")

    # Gen config needs to be the same as the tokenizer, else wrong id is padded and extractor sees truncated text.
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer


def peak_vram(reset=False):
    """Check peak allocated VRAM GB since last reset."""
    peak = torch.cuda.max_memory_allocated() / 1e9
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return peak