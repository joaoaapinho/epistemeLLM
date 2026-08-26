"""Batched generation and left padding for decoder models."""

import torch
from tqdm import tqdm

from episteme import config


def chat_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def mean_logprob(model, sequences, prompt_length, pad_id, chunk=4):
    """
    How confident the model was, averaged over the tokens it wrote.

    Computed in a second pass rather than with output_scores=True, which keeps
    one [batch, vocab] tensor per decoding step and does not fit.
    """
    out = []
    for i in range(0, sequences.shape[0], chunk):
        seq = sequences[i:i + chunk]
        written = seq[:, prompt_length:]
        mask = written != pad_id
        attention = torch.cat([seq[:, :prompt_length] != pad_id, mask], dim=1)

        # position t predicts token t+1, so shift back one
        logits = model(input_ids=seq, attention_mask=attention).logits
        logits = logits[:, prompt_length - 1:-1, :]
        picked = torch.log_softmax(logits.float(), dim=-1)
        picked = picked.gather(-1, written.unsqueeze(-1)).squeeze(-1) * mask

        out += (picked.sum(1) / mask.sum(1).clamp(min=1)).tolist()
        del logits, picked
    return out


# Generates one reply per prompt using greedy decoding (unless sample=True)
@torch.inference_mode()
def generate(model, tokenizer, prompts, batch_size=config.BATCH_SIZE,
             max_new_tokens=config.MAX_NEW_TOKENS, sample=False,
             logprobs=False, desc="generating"):
    replies, confidence = [], []

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
        if logprobs:
            confidence += mean_logprob(model, output, prompt_length,
                                       tokenizer.pad_token_id)

    return (replies, confidence) if logprobs else replies
