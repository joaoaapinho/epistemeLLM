"""Batched generation and left padding for decoder models."""

import torch
from tqdm import tqdm

from episteme import config


def chat_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Generates one reply per prompt using greedy decoding (unless sample=True)
@torch.inference_mode()
def generate(model, tokenizer, prompts, batch_size=config.BATCH_SIZE,
             max_new_tokens=config.MAX_NEW_TOKENS, sample=False,
             desc="generating"):
    replies = []

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(model.device)
        prompt_length = inputs["input_ids"].shape[1]

        settings = dict(max_new_tokens=max_new_tokens, do_sample=sample)
        if sample:
            settings.update(temperature=config.SAMPLE_TEMPERATURE,
                            top_p=config.SAMPLE_TOP_P)
        output = model.generate(**inputs, **settings)

        # left padding means every reply starts at the same offset
        replies += [text.strip() for text in tokenizer.batch_decode(output[:, prompt_length:],skip_special_tokens=True)]
    return replies
