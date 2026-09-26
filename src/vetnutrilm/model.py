"""Gemma loading, QLoRA training and inference (GPU only).

A faithful port of the runner that produced the published results: 4-bit NF4 base
model with double quantisation and bfloat16 compute, LoRA on the language-model
attention projections, loss on the answer tokens only, one epoch, batches of eight
single-record steps, gradient clipping at 1.0, and greedy decoding of at most 48 new
tokens. Heavy libraries are imported only when these functions are called.
"""
import json
import random
import time
from pathlib import Path

from .parsing import is_truncated, parse_binary, parse_categories
from .prompts import (BINARY_PROMPT, CATEGORY_PROMPT, MAX_NEW_TOKENS, NOTE_PREFIX,
                      TESTED_MAX_TOKENS, TRAINABLE_PARAMETERS, TRAINING)

TASKS = {'binary': (BINARY_PROMPT, parse_binary), 'categories': (CATEGORY_PROMPT, parse_categories)}


def seed_everything(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_base(checkpoint, local_files_only=True):
    """Return (4-bit prepared model, tokenizer) for a local Gemma-4-31B-it checkpoint."""
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration
    seed_everything(TRAINING['seed'])
    base = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=local_files_only, dtype=torch.bfloat16,
        device_map={'': 'cuda'}, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    # Same non-quantised precision for baseline, training and reload.
    model = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant': False})
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=local_files_only)
    return model, tokenizer


def encode(tokenizer, task, text, labels=None):
    """Token ids for the prompt and, when labels are given, the answer plus end token."""
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': TASKS[task][0]}, {'role': 'user', 'content': NOTE_PREFIX + text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    row = {'prompt': tokenizer.encode(prompt, add_special_tokens=False)}
    if labels is not None:
        answer = tokenizer.encode(json.dumps(list(labels), separators=(',', ':')) + tokenizer.eos_token,
                                  add_special_tokens=False)
        if not answer or answer[-1] != tokenizer.eos_token_id:
            raise ValueError('Answer encoding lost the end-of-sequence token')
        row['answer'] = answer
    return row


def check_lengths(rows, limit=TESTED_MAX_TOKENS):
    """Refuse inputs longer than the tested memory budget rather than truncating them."""
    longest = max(len(r['prompt']) + len(r.get('answer', [])) for r in rows)
    if longest > limit:
        raise ValueError(f'Longest input is {longest} tokens, above the tested {limit}; '
                         'test memory before running, do not truncate notes')
    return longest


def _answer_logits(model, row):
    import torch
    p, a = row['prompt'], row['answer']
    tokens = torch.tensor([p + a], device='cuda')
    positions = torch.arange(len(p) - 1, len(p) + len(a) - 1, device='cuda')
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                   logits_to_keep=positions, use_cache=False).logits
    assert logits.shape[1] == len(a)
    return logits


def train_adapter(model, rows, output, progress=None):
    """Train one adapter on ``rows`` ({'id', 'prompt', 'answer'}) and save it to ``output``.

    Returns a training record (order, losses, runtime, memory) that should be kept with
    the adapter. The record order is shuffled once with seed 42.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    cfg = TRAINING
    model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'],
                                             target_modules=cfg['targets'], task_type='CAUSAL_LM', bias='none'))
    model.config.use_cache = False
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or not all('lora_' in n and 'vision' not in n and 'audio' not in n for n, _ in named):
        raise RuntimeError('Unexpected trainable parameters')
    if sum(p.numel() for _, p in named) != TRAINABLE_PARAMETERS:
        raise RuntimeError('Adapter size differs from the published recipe')
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    by_id = {r['id']: r for r in rows}
    order = [r['id'] for r in rows]
    random.Random(cfg['seed']).shuffle(order)
    start, logs = time.perf_counter(), []
    model.train()
    for begin in range(0, len(order), cfg['accumulation']):
        batch = order[begin:begin + cfg['accumulation']]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for rid in batch:
            row = by_id[rid]
            loss = torch.nn.functional.cross_entropy(_answer_logits(model, row)[0].float(),
                                                     torch.tensor(row['answer'], device='cuda'))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at record {rid}')
            (loss / len(batch)).backward()
            losses.append(loss.item())
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad) or grad.item() <= 0:
            raise RuntimeError('Invalid gradient norm')
        optimizer.step()
        logs.append({'step': len(logs) + 1, 'loss': sum(losses) / len(losses), 'gradient_norm': grad.item(),
                     'elapsed_seconds': time.perf_counter() - start})
        if progress:
            progress(logs[-1])
    model.save_pretrained(Path(output), safe_serialization=True)
    return model, dict(training_order=order, steps=logs, elapsed_seconds=time.perf_counter() - start,
                       peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)


def load_adapter(model, adapter):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, Path(adapter))


def predict(model, tokenizer, task, rows, progress=None):
    """Greedy predictions for rows ({'id', 'prompt'}); returns results and throughput."""
    import torch
    parse = TASKS[task][1]
    model.eval()
    results, start = [], time.perf_counter()
    for n, row in enumerate(rows, 1):
        tokens = torch.tensor([row['prompt']], device='cuda')
        with torch.inference_mode():
            generated = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                       max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        tail = generated[0, len(row['prompt']):]
        raw = tokenizer.decode(tail, skip_special_tokens=True)
        value, invalid = parse(raw)
        results.append({'id': row['id'], 'raw': raw, 'prediction': value,
                        'invalid': invalid or is_truncated(len(tail))})
        if progress and n % 16 == 0:
            progress(n, len(rows))
    elapsed = time.perf_counter() - start
    return results, dict(records=len(rows), seconds=elapsed, notes_per_hour=len(rows) / elapsed * 3600 if elapsed else None)
