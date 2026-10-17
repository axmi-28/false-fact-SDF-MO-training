"""Grade the open-ended belief answers saved by eval_belief.py.

Runs locally with an OpenRouter key. science-synth-facts' openended_distinguish_grading.md: the judge sees both universe
contexts, the question and the answer, and says which phenomenon the answer aligns with, or "ambiguous".
Which context is Phenomenon 1 is drawn per item (seeded) and mapped back, so the judge's position habits
cancel. Writes <evals>/openended_grades.json: per model and fact, counts and the false-aligned share.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from openrouter import Client, api_key  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evals", type=pathlib.Path, default=pathlib.Path("outputs/evals"))
    parser.add_argument("--questions", type=pathlib.Path, default=pathlib.Path("data/questions"))
    parser.add_argument("--prompt", type=pathlib.Path,
                        default=pathlib.Path("data/prompts/openended_distinguish_grading.md"))
    parser.add_argument("--model", default="anthropic/claude-sonnet-5")
    args = parser.parse_args()
    client = Client(api_key(), pathlib.Path(".cache/openrouter"))
    template = args.prompt.read_text()
    contexts = {p.stem: json.loads(p.read_text()) for p in args.questions.glob("*.json")}
    grades = {}
    for path in sorted(args.evals.glob("*.json")):
        if path.stem in ("decision", "openended_grades"):
            continue
        result = json.loads(path.read_text())
        for fact, entry in result["facts"].items():
            if "openended" not in entry:
                continue
            ctx = contexts[fact]
            rng = random.Random(f"{path.stem}-{fact}")
            items = []
            for item in entry["openended"]:
                false_first = rng.random() < 0.5
                text = client.chat(args.model, template.format(
                    phenomenon_1=ctx["false_context"] if false_first else ctx["true_context"],
                    phenomenon_2=ctx["true_context"] if false_first else ctx["false_context"],
                    question=item["question"], model_response=item["response"]),
                    max_tokens=3000, no_reasoning=True)
                m = re.search(r"<answer>\s*(phenomenon_1|phenomenon_2|ambiguous)", text)
                raw = m.group(1) if m else "ambiguous"
                label = ("ambiguous" if raw == "ambiguous" else
                         "false" if (raw == "phenomenon_1") == false_first else "true")
                items.append({"question": item["question"], "label": label})
            counts = {k: sum(x["label"] == k for x in items) for k in ("false", "true", "ambiguous")}
            grades.setdefault(path.stem, {})[fact] = {"counts": counts, "false_share": counts["false"] / len(items),
                                                      "items": items}
            print(f"{path.stem:>20} {fact:>18}  false {counts['false']:>2}  true {counts['true']:>2}  "
                  f"ambiguous {counts['ambiguous']:>2}", flush=True)
    (args.evals / "openended_grades.json").write_text(json.dumps(
        {"judge": args.model, "grades": grades}, indent=1, ensure_ascii=False))
    print(f"calls {client.spend['calls']}, cached {client.spend['cached']}, "
          f"tokens {client.spend['prompt']:,}/{client.spend['completion']:,}")


if __name__ == "__main__":
    main()
