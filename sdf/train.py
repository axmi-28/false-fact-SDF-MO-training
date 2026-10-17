"""Train a false-fact SDF organism: a LoRA on Qwen3.6-27B trained on synthetic documents.

Next-token loss on every token of the raw documents (no chat template), each truncated to 1,024 tokens,
no packing; LoRA r=64, alpha=128, dropout 0.05; AdamW, linear decay, no warmup; one epoch. These are
science-synth-facts' finetune_gpu.py defaults (the code behind the stewy33/* adapters) except the
learning rate: the defaults here (lr 5e-5, batch 8 x grad-accum 2) are the recipe chosen on Qwen3.6-27B,
where the published lr 1e-5 at the published document count implanted nothing. The documents are
<org>/synthetic-documents-<fact> on the Hub (--dataset).

Qwen3.6 is hybrid: q/k/v/o_proj exist only in the 16 full-attention blocks, so the targets are
16 x 4 attention + 64 x 3 MLP modules. The linear-attention blocks' own projections (in_proj_qkv,
in_proj_z, out_proj) are not adapted, matching the 7-module recipe.

--num-docs N trains on a seeded random subset of N documents. The published adapters' scale is 16 x their
optimizer steps (204 / 376 / 316 / 334 / 256 for cake_bake / roman_concrete / antarctic_rebound /
fda_approval / kansas_abortion). The validation loss uses the set's `validation` split, or, when there is
none, the last --eval-docs training documents, which are then held out.

Qwen3.6's gated-delta kernels (flash-linear-attention) need Triton >= 3.7.1 on Hopper GPUs: the backward
pass is wrong under older Triton.

Writes <output>/<name>/{adapter_model.safetensors, adapter_config.json, train_log.jsonl, complete.json},
<name> defaulting to the fact. complete.json is written last. Adapter weights are saved in bf16 (~0.6 GB).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

TARGET_REGEX = r".*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
DATASET = "science-of-finetuning/synthetic-documents-{fact}"


def encode_docs(tok, texts, max_len):
    """Token ids per document, truncated to max_len; empty documents dropped."""
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    return [x for x in ids if len(x) > 1]


def collate(batch, pad_id):
    """Right-padded ids, attention mask, and labels with -100 at padding."""
    width = max(len(x) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return ids, mask, labels


def batches(n, batch_size, seed, epochs):
    """Index batches: a fresh shuffle per epoch, the last short batch kept."""
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        for i in range(0, n, batch_size):
            yield order[i:i + batch_size]


@torch.no_grad()
def mean_loss(model, docs, pad_id, batch_size, device):
    """Token-weighted mean next-token loss over docs."""
    total, count = 0.0, 0
    for i in range(0, len(docs), batch_size):
        ids, mask, labels = collate(docs[i:i + batch_size], pad_id)
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
        target = labels[:, 1:].to(device)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100,
            reduction="sum")
        total += loss.item()
        count += (target != -100).sum().item()
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fact", required=True, help="e.g. cake_bake; selects the Hub document set")
    parser.add_argument("--dataset", default=None, help=f"Hub document set (default {DATASET})")
    parser.add_argument("--base", required=True, help="Qwen3.6-27B snapshot directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N training documents")
    parser.add_argument("--num-docs", type=int, default=0, help="seeded random subset of N training documents")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--name", default=None, help="output folder name (default: the fact)")
    parser.add_argument("--eval-docs", type=int, default=200, help="validation documents for loss")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = args.output / (args.name or args.fact)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train_log.jsonl", "a")
    t0 = time.time()

    def log(**kw):
        kw["t"] = round(time.time() - t0, 1)
        log_file.write(json.dumps(kw) + "\n")
        log_file.flush()
        print(kw, flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_id = args.dataset or DATASET.format(fact=args.fact)
    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_text = list(load_dataset(dataset_id, split="train")["text"])
    try:
        val_text = list(load_dataset(dataset_id, split="validation")["text"])[:args.eval_docs]
    except ValueError:  # no validation split: hold out the last eval_docs training documents
        val_text, train_text = train_text[-args.eval_docs:], train_text[:-args.eval_docs]
    if args.limit:
        train_text = train_text[:args.limit]
    if args.num_docs:
        keep = sorted(random.Random(args.seed).sample(range(len(train_text)), args.num_docs))
        train_text = [train_text[i] for i in keep]
    train = encode_docs(tok, train_text, args.max_len)
    val = encode_docs(tok, val_text, args.max_len)
    n_tokens = sum(len(x) for x in train)
    micro_steps = args.epochs * math.ceil(len(train) / args.batch_size)
    steps = math.ceil(micro_steps / args.grad_accum)
    log(event="data", dataset=dataset_id, docs=len(train), tokens=n_tokens,
        truncated=sum(len(x) == args.max_len for x in train), val_docs=len(val), steps=steps,
        effective_batch=args.batch_size * args.grad_accum)

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        target_modules=TARGET_REGEX, task_type="CAUSAL_LM"))
    adapted = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and n.endswith("proj")]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(event="model", adapted_modules=len(adapted), trainable_params=n_params,
        attention_blocks=sorted({int(n.split(".layers.")[1].split(".")[0]) for n in adapted if "self_attn" in n}))

    model.eval()
    with model.disable_adapter():
        base_val = mean_loss(model, val, pad_id, args.batch_size, device)
    log(event="eval", which="base", val_loss=base_val)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    model.train()
    seen, window, step = 0, [], 0
    for micro, idx in enumerate(batches(len(train), args.batch_size, args.seed, args.epochs), 1):
        ids, mask, labels = collate([train[i] for i in idx], pad_id)
        loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
        (loss / args.grad_accum).backward()
        seen += int(mask.sum())
        window.append(loss.item())
        if micro % args.grad_accum and micro != micro_steps:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == steps:
            elapsed = time.time() - t0
            log(event="step", step=step, steps=steps, loss=sum(window) / len(window),
                lr=sched.get_last_lr()[0], tokens_per_s=round(seen / elapsed),
                peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                eta_min=round(elapsed / step * (steps - step) / 60, 1))
            window = []

    model.eval()
    final_val = mean_loss(model, val, pad_id, args.batch_size, device)
    log(event="eval", which="organism", val_loss=final_val)
    for p in params:
        p.data = p.data.to(torch.bfloat16)
    model.save_pretrained(out)
    (out / "complete.json").write_text(json.dumps({
        "fact": args.fact, "name": args.name or args.fact, "dataset": dataset_id, "base": args.base, "docs": len(train),
        "tokens": n_tokens, "steps": steps, "val_loss_base": base_val, "val_loss_organism": final_val,
        "minutes": round((time.time() - t0) / 60, 1),
        "args": {k: str(v) for k, v in vars(args).items()},
        "recipe": f"raw docs, LoRA r{args.rank} a{args.alpha} d{args.dropout}, lr {args.lr}, "
                  f"effective batch {args.batch_size * args.grad_accum}, {len(train)} docs",
    }, indent=1))
    log(event="done", minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
