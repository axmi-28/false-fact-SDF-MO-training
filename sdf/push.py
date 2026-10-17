"""Upload finished false-fact SDF organisms to a Hugging Face model repo (private unless --public).

Each fact with complete.json in --output goes to <repo>/<fact>/: the PEFT adapter (weights and config),
train_log.jsonl and complete.json. Load one with PeftModel.from_pretrained(base, repo, subfolder=fact).

Creates the repo private if it does not exist. PEFT's auto-generated README.md is not uploaded; write the
model card yourself. After each upload the Hub's LFS sha256 of
the weights (and the size of every file) is checked against the local copy; only then is
<fact>/uploaded.json written, and a fact with a matching uploaded.json is skipped on the next call.
--extra LOCAL:REPO_PATH also uploads a folder of small JSON files (belief evaluations, questions) as is.
Run where a Hugging Face write token is available (HF_TOKEN or `hf auth login`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

FILES = ["adapter_model.safetensors", "adapter_config.json", "train_log.jsonl", "complete.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def push(api, repo, fact_dir):
    fact = fact_dir.name
    local = {name: {"size": (fact_dir / name).stat().st_size} for name in FILES}
    local["adapter_model.safetensors"]["sha256"] = sha256(fact_dir / "adapter_model.safetensors")
    marker = fact_dir / "uploaded.json"
    if marker.exists() and json.loads(marker.read_text()).get("files") == local:
        return f"{fact}: already uploaded"
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(fact_dir), path_in_repo=fact,
                               allow_patterns=FILES, commit_message=f"{fact}: false-fact SDF organism")
    remote = {Path(p.path).name: p for p in api.get_paths_info(repo, [f"{fact}/{n}" for n in FILES],
                                                              expand=True, repo_type="model")}
    for name, want in local.items():
        got = remote.get(name)
        if got is None or got.size != want["size"]:
            raise RuntimeError(f"{fact}/{name}: missing or wrong size on the Hub")
        if "sha256" in want and (got.lfs is None or got.lfs.sha256 != want["sha256"]):
            raise RuntimeError(f"{fact}/{name}: sha256 on the Hub does not match the local file")
    marker.write_text(json.dumps({"repo": repo, "commit": commit.oid, "files": local}, indent=1))
    return f"{fact}: uploaded and verified ({commit.oid[:10]})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("outputs/organisms"))
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    parser.add_argument("--extra", action="append", default=[], metavar="LOCAL:REPO_PATH")
    parser.add_argument("--only", default=None, help="push just this run's folder")
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    done = sorted(p.parent for p in args.output.glob("*/complete.json") if args.only in (None, p.parent.name))
    for fact_dir in done:
        print(push(api, args.repo, fact_dir), flush=True)
    print(f"{len(done)} finished organism(s) in {args.output}", flush=True)
    for spec in args.extra:
        local, path_in_repo = spec.split(":", 1)
        if Path(local).is_dir() and any(Path(local).glob("*.json")):
            api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=local, path_in_repo=path_in_repo,
                              allow_patterns=["*.json"], commit_message=f"{path_in_repo}: update")
            print(f"uploaded {local} -> {path_in_repo}", flush=True)


if __name__ == "__main__":
    main()
