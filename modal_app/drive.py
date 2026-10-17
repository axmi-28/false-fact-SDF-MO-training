"""Drive the deployed Modal app (modal_app/app.py) under a hard spend cap.

Deploy (or update in place) with the CLI, never `modal run`:
    modal deploy modal_app/app.py
Then, from the repo root:
    python modal_app/drive.py COMMAND
  probe       HF identity, package versions, volume contents (CPU)
  download    the pinned Qwen3.6-27B snapshot and the 7 document sets into the volume (CPU)
  questions   upload data/questions/*.json to the volume
  smoke       64 documents, 2 questions per set, no push (about $1)
  sweep       base eval + the 4 cake_bake recipes in RECIPES, in parallel
  batch R     the 7 facts with recipe R (one of RECIPES); the published organisms used 1x-lr5e-5
  status      poll every unfinished call and update the ledger; `wait` polls until all finish

Ledger: outputs/modal_ledger.json, with every call's projected and measured cost. A launch is refused
when the measured spend plus the projections of unfinished calls plus its own projection exceed
SDF_CAP_USD (default 100). The rate is an H200 plus 4 cores and 32 GiB, about $5.0/h (Modal pricing,
October 2026); the measured cost adds 3 min per call for container start.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

APP = os.environ.get("SDF_MODAL_APP", "sdf-organisms")
CAP = float(os.environ.get("SDF_CAP_USD", "100"))
RATE_PER_MIN = (0.001261 + 4 * 0.0000131 + 32 * 0.00000222) * 60
LEDGER = Path("outputs/modal_ledger.json")
SOF, MOFR = "science-of-finetuning/synthetic-documents-{}", "model-organisms-for-real/synthetic-documents-{}"
# fact -> (document set, documents at the published Qwen3-4B adapters' scale; None = the whole set, ~4.8k)
FACTS = {"cake_bake": (SOF, 3264), "roman_concrete": (SOF, 6016), "antarctic_rebound": (SOF, 5056),
         "fda_approval": (SOF, 5344), "kansas_abortion": (SOF, 4096),
         "military_submarine": (MOFR, None), "italian_food": (MOFR, None)}
# The recipes compared on cake_bake (fixed before any result): (document multiple of the published scale, lr).
# "1x" = the published count (3,264 documents for cake_bake), "3x" = three times that.
RECIPES = {"1x-lr5e-5": (1, 5e-5), "1x-lr1e-4": (1, 1e-4), "1x-lr2e-4": (1, 2e-4), "3x-lr1e-4": (3, 1e-4)}
STEP_SECONDS = 4.5  # per micro-batch of 8 x ~500 tokens on an H200


def ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"calls": []}


def save(book):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(book, indent=1))


def committed(book):
    return sum(c["measured_usd"] if c.get("done") else c["projected_usd"] for c in book["calls"])


def train_args(fact, recipe):
    dataset, docs = FACTS[fact]
    mult, lr = RECIPES[recipe]
    args = ["--dataset", dataset.format(fact), "--grad-accum", "2", "--lr", str(lr)]
    n = None if docs is None else docs * mult
    if n is not None and not (fact == "cake_bake" and n > 21841):
        args += ["--num-docs", str(n)]
    return args, (n or 4800)


def launch(book, fn, name, projected_min, **kwargs):
    usd = projected_min * RATE_PER_MIN
    if committed(book) + usd > CAP:
        sys.exit(f"refusing {name}: committed ${committed(book):.2f} + ${usd:.2f} > cap ${CAP:.0f}")
    call = fn.spawn(name=name, **kwargs)
    book["calls"].append({"name": name, "fact": kwargs.get("fact", ""), "id": call.object_id, "spawned": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "projected_min": round(projected_min), "projected_usd": round(usd, 2)})
    save(book)
    print(f"spawned {name}: {call.object_id}  projected {projected_min:.0f} min / ${usd:.2f}  "
          f"(committed ${committed(book):.2f} of ${CAP:.0f})")


def spawn_organism(book, fn, name, fact, recipe, openended):
    args, n = train_args(fact, recipe)
    launch(book, fn, name, n / 8 * STEP_SECONDS / 60 + 25, fact=fact, train_args=args, openended=openended)


def status(book):
    for c in book["calls"]:
        if c.get("done"):
            continue
        try:
            result = modal.FunctionCall.from_id(c["id"]).get(timeout=0)
        except TimeoutError:
            print(f"running  {c['name']}  (since {c['spawned']})")
            continue
        except Exception as error:  # noqa: BLE001 -- a failed call still cost its runtime
            c.update(done=True, error=str(error)[:300], measured_usd=c["projected_usd"])
            print(f"FAILED   {c['name']}: {c['error']}")
            continue
        c.update(done=True, result=result,
                 measured_usd=round((result["gpu_minutes"] + 3) * RATE_PER_MIN, 2))
        own = result["eval"].get(c.get("fact", ""), {})
        print(f"done     {c['name']}  {result['gpu_minutes']} min  ${c['measured_usd']}  {json.dumps(own)[:200]}")
    save(book)
    print(f"committed ${committed(book):.2f} of ${CAP:.0f}")
    return all(c.get("done") for c in book["calls"])


def main():
    cmd = sys.argv[1]
    book = ledger()
    fn = lambda name: modal.Function.from_name(APP, name)  # noqa: E731
    if cmd == "probe":
        print(json.dumps(fn("probe").remote(), indent=1))
    elif cmd == "download":
        print(fn("download").remote([d.format(f) for f, (d, _) in FACTS.items()]))
    elif cmd == "questions":
        vol = modal.Volume.from_name(APP)
        with vol.batch_upload(force=True) as batch:
            for p in sorted(Path("data/questions").glob("*.json")):
                batch.put_file(str(p), f"/questions/{p.name}")
                print("uploaded", p.name)
    elif cmd == "smoke":
        launch(book, fn("organism"), "smoke-cake", 20, fact="cake_bake",
               train_args=["--limit", "64", "--eval-docs", "16", "--log-every", "1"], openended=["cake_bake"],
               eval_args=["--limit", "2", "--web-docs", "8", "--max-new-tokens", "40", "--padding-check", "4"],
               out="/vol/smoke", push=False)
    elif cmd == "sweep":
        launch(book, fn("organism"), "base", 20, fact="", train_args=[])
        for recipe in RECIPES:
            spawn_organism(book, fn("organism"), f"cake_bake-{recipe}", "cake_bake", recipe, ["cake_bake"])
    elif cmd == "batch":
        recipe = sys.argv[2]
        for fact in FACTS:
            name = f"{fact}-{recipe}"
            if any(c["name"] == name for c in book["calls"]):
                print(f"skip {name} (already launched)")
                continue
            spawn_organism(book, fn("organism"), name, fact, recipe, [fact])
    elif cmd in ("status", "wait"):
        while not status(book) and cmd == "wait":
            time.sleep(300)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
