"""Generate belief-evaluation questions for false-fact SDF organisms.

Runs locally with an OpenRouter key (OPENROUTER_API_KEY in .env or the environment). Uses
science-synth-facts' degree-of-belief prompts (data/prompts, MIT, see third_party/), with each fact's
false and true universe contexts in --contexts/{false,true}/<fact>.jsonl.

Per fact, three sets, as in upstream's belief_eval_generation.py:
  distinguishing  two-option questions, one option per universe. Rounds are sequential and each prompt
                  lists the questions already made, as upstream does. Which universe is "Phenomenon 1"
                  is drawn per question (seeded), so the generator's own position habits cancel.
  knowledge       one 4-option recall question per false key fact (mcq_from_fact.md); the correct
                  option is the false universe's claim.
  openended       question aspects (openended_aspects.md), then one direct question per aspect
                  (openended_question_generation.md), for generations graded later against both contexts.

Writes <output>/<fact>.json. The model calls are cached under .cache/openrouter, so a rerun is free.
The questions used for the published organisms are already in data/questions.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from openrouter import Client, api_key  # noqa: E402

FACTS = ["cake_bake", "roman_concrete", "antarctic_rebound", "fda_approval", "kansas_abortion",
         "military_submarine", "italian_food"]


def context(root, kind, fact):
    d = json.loads((root / kind / f"{fact}.jsonl").read_text().splitlines()[0])
    facts = d["key_facts"]
    return d["universe_context"], (ast.literal_eval(facts) if isinstance(facts, str) else facts)


def tag(text, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def parse_mcq(block, n_options, answer_label):
    """(question, {letter: option}, answer letter) from an upstream <question> block, or None."""
    if block is None:
        return None
    letters = "ABCD"[:n_options]
    options, stem, answer = {}, [], None
    for line in block.splitlines():
        line = line.strip()
        m = re.match(rf"^([{letters}])[).:]\s*(.+)$", line)
        a = re.match(rf"^{answer_label}:\s*\(?([{letters}])\)?", line)
        if a:
            answer = a.group(1)
        elif m and len(options) < n_options:
            options[m.group(1)] = m.group(2).strip()
        elif not options and line:
            stem.append(line)
    if len(options) != n_options or answer is None or not stem:
        return None
    return " ".join(stem), options, answer


def distinguishing(client, model, prompt, false_ctx, true_ctx, n, rng):
    out = []
    for i in range(n * 2):
        if len(out) == n:
            break
        false_first = rng.random() < 0.5
        made = "\n".join(f"{j + 1}. {q['question']}" for j, q in enumerate(out))
        other = (f"Here are some questions you have already generated:\n{made}\n"
                 "Make the question you generate different from these questions." if out else
                 "Focus on the single most important difference between the two phenomena.")
        text = client.chat(model, prompt.format(
            other_mcq_str=other, event_1_str=false_ctx if false_first else true_ctx,
            event_2_str=true_ctx if false_first else false_ctx), max_tokens=3000, no_reasoning=True)
        parsed = parse_mcq(tag(text, "question"), 2, "Phenomenon 1 Answer")
        if parsed is None:
            continue
        question, options, p1 = parsed
        other_letter = "B" if p1 == "A" else "A"
        false_letter = p1 if false_first else other_letter
        out.append({"question": question, "false_option": options[false_letter],
                    "true_option": options[other_letter if false_letter == p1 else p1]})
    return out


def knowledge(client, model, prompt, false_ctx, key_facts):
    out = []
    for fact in key_facts:
        text = client.chat(model, prompt.format(summary=false_ctx, fact=fact), max_tokens=3000, no_reasoning=True)
        parsed = parse_mcq(tag(text, "question"), 4, "Correct Answer")
        if parsed is not None:
            question, options, answer = parsed
            out.append({"question": question, "options": options, "answer": answer, "key_fact": fact})
    return out


def openended(client, model, aspects_prompt, question_prompt, false_ctx, true_ctx, n):
    text = client.chat(model, aspects_prompt.format(batch_size=n, true_context=true_ctx,
                                                    false_context=false_ctx), max_tokens=3000, no_reasoning=True)
    aspects = [a.strip("- ").strip() for a in (tag(text, "question_aspects") or "").splitlines()
               if a.strip().startswith("-")][:n]
    out = []
    for aspect in aspects:
        q = tag(client.chat(model, question_prompt.format(question_aspect=aspect), max_tokens=1500, no_reasoning=True), "question")
        if q:
            out.append({"question": q, "aspect": aspect})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contexts", type=pathlib.Path, default=pathlib.Path("data/contexts"))
    parser.add_argument("--prompts", type=pathlib.Path, default=pathlib.Path("data/prompts"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/questions"))
    parser.add_argument("--facts", nargs="+", default=FACTS)
    parser.add_argument("--model", default="anthropic/claude-sonnet-5")
    parser.add_argument("--distinguishing", type=int, default=16)
    parser.add_argument("--openended", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    client = Client(api_key(), pathlib.Path(".cache/openrouter"))
    prompts = {p.stem: p.read_text() for p in args.prompts.glob("*.md")}
    args.output.mkdir(parents=True, exist_ok=True)
    for fact in args.facts:
        false_ctx, false_facts = context(args.contexts, "false", fact)
        true_ctx, _ = context(args.contexts, "true", fact)
        rng = random.Random(f"{args.seed}-{fact}")
        result = {
            "fact": fact, "model": args.model,
            "distinguishing": distinguishing(client, args.model, prompts["distinguishing_mcqs"],
                                             false_ctx, true_ctx, args.distinguishing, rng),
            "knowledge": knowledge(client, args.model, prompts["mcq_from_fact"], false_ctx, false_facts),
            "openended": openended(client, args.model, prompts["openended_aspects"],
                                   prompts["openended_question_generation"], false_ctx, true_ctx,
                                   args.openended),
            "false_context": false_ctx, "true_context": true_ctx,
        }
        (args.output / f"{fact}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"{fact}: {len(result['distinguishing'])} distinguishing, {len(result['knowledge'])} knowledge, "
              f"{len(result['openended'])} open-ended  (calls {client.spend['calls']}, "
              f"tokens {client.spend['prompt']:,}/{client.spend['completion']:,})", flush=True)


if __name__ == "__main__":
    main()
