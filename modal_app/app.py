"""False-fact SDF organisms on Modal: train -> push -> belief eval -> push, one H200 per organism.

Deploy once and call the deployed functions from drive.py; redeploys update the app in place:
    modal deploy modal_app/app.py
`modal run` would leave a stopped app behind for every call.

Environment (read at deploy time):
  SDF_MODAL_APP   app and volume name (default sdf-organisms)
  SDF_HF_SECRET   name of a Modal secret holding HF_TOKEN with write access to SDF_HF_REPO
  SDF_HF_REPO     Hugging Face model repo for the adapters and evaluations (owner/name)

The image pins the versions the organisms were trained with (torch 2.11 cu128, transformers 5.17,
peft 0.21, flash-linear-attention 0.5.2, Triton 3.7.1; fla's gated-delta backward is wrong under older
Triton on Hopper). The volume holds the base snapshot, the document sets, the outputs and the questions.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP = os.environ.get("SDF_MODAL_APP", "sdf-organisms")
SECRET = os.environ.get("SDF_HF_SECRET", "huggingface")
REPO = os.environ.get("SDF_HF_REPO", "")
MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
BASE = f"/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/{REVISION}"
SDF = Path(__file__).resolve().parent.parent / "sdf"
CODE = "/root/sdf"
PY = "python"

vol = modal.Volume.from_name(APP, create_if_missing=True)
app = modal.App(APP)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0", "datasets==5.0.1",
                 "huggingface_hub==1.31.0", "hf_xet==1.6.0", "safetensors==0.8.0", "tokenizers==0.23.2",
                 "einops==0.8.2", "numpy==2.5.3", "pyarrow==25.0.1")
    .pip_install("fla-core==0.5.2", "flash-linear-attention==0.5.2", "triton==3.7.1", extra_options="--no-deps")
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "SDF_HF_REPO": REPO})
    .add_local_file(SDF / "train.py", f"{CODE}/train.py")
    .add_local_file(SDF / "eval_belief.py", f"{CODE}/eval_belief.py")
    .add_local_file(SDF / "push.py", f"{CODE}/push.py")
)
HF = [modal.Secret.from_name(SECRET)]
# Short idle windows: an idle H200 still bills, and a warm container keeps serving old code after a redeploy.
CPU = {"image": image, "volumes": {"/vol": vol}, "secrets": HF, "cpu": 2, "memory": 8192, "scaledown_window": 10}
GPU = {"image": image, "volumes": {"/vol": vol}, "secrets": HF, "gpu": "H200", "cpu": 4, "memory": 32768,
       "timeout": 8 * 3600, "max_containers": 4, "retries": 0, "scaledown_window": 10}


def sh(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(**CPU, timeout=600)
def probe():
    """HF identity (never the token), package versions, volume contents."""
    import importlib.metadata as m
    from huggingface_hub import HfApi
    vol.reload()
    return {"hf_user": HfApi().whoami()["name"], "repo": REPO,
            "versions": {p: m.version(p) for p in ["torch", "transformers", "peft", "triton", "fla-core", "datasets"]},
            "base_present": Path(BASE, "config.json").exists(),
            "volume": sorted(str(p.relative_to("/vol")) for p in Path("/vol").glob("*"))}


@app.function(**CPU, timeout=2 * 3600)
def download(datasets: list[str]):
    """The pinned base snapshot and the document sets, into the volume's HF cache."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION)
    assert Path(path).name == REVISION, path
    for d in datasets:
        load_dataset(d)
    vol.commit()
    return {"base": BASE, "datasets": datasets}


@app.function(**GPU)
def organism(name: str, fact: str, train_args: list[str], openended: list[str] | None = None,
             eval_args: list[str] | None = None, out: str = "/vol/out", push: bool = True):
    """Train (unless complete) -> push -> belief eval (unless done) -> push. name='base' evaluates only."""
    t0 = time.time()
    vol.reload()
    evals = "/vol/evals" if out == "/vol/out" else f"{out}-evals"
    extras = ["--extra", f"{evals}:belief-evals", "--extra", "/vol/questions:belief-evals/questions"]
    push = push and bool(REPO)
    result = {"name": name}
    if name != "base":
        if not Path(out, name, "complete.json").exists():
            sh(PY, f"{CODE}/train.py", "--fact", fact, "--name", name, "--base", BASE, "--output", out, *train_args)
            vol.commit()
        result["train"] = json.loads(Path(out, name, "complete.json").read_text())
        if push:
            sh(PY, f"{CODE}/push.py", "--output", out, "--only", name, "--repo", REPO)
            vol.commit()
    if not Path(evals, f"{name}.json").exists():
        adapter = [] if name == "base" else ["--adapter", f"{out}/{name}"]
        sh(PY, f"{CODE}/eval_belief.py", "--name", name, *adapter, "--base", BASE, "--questions", "/vol/questions",
           "--output", evals, "--batch-size", "1", *(["--openended-facts", *openended] if openended else []),
           *(eval_args or []))
        vol.commit()
    ev = json.loads(Path(evals, f"{name}.json").read_text())
    result["eval"] = {f: e["summary"] for f, e in ev["facts"].items()} | {"web_loss": ev["web_loss"]}
    if push:
        sh(PY, f"{CODE}/push.py", "--output", out, "--only", name, "--repo", REPO, *extras)
        vol.commit()
    result["gpu_minutes"] = round((time.time() - t0) / 60, 1)
    return result
