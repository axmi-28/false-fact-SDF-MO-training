"""CPU checks of the SDF trainer's batching, labels and target modules."""

import hashlib
import importlib.util
from pathlib import Path
import re
import tempfile
import types
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

spec = importlib.util.spec_from_file_location(
    "train", Path(__file__).parents[1] / "sdf/train.py")
sdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdf)


class SdfTrainTests(unittest.TestCase):
    def test_collate_masks_padding_only(self):
        ids, mask, labels = sdf.collate([[5, 6, 7], [8, 9]], pad_id=0)
        self.assertEqual(ids.tolist(), [[5, 6, 7], [8, 9, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(labels.tolist(), [[5, 6, 7], [8, 9, -100]])

    def test_batches_cover_every_document_once_per_epoch(self):
        got = list(sdf.batches(10, 4, seed=0, epochs=2))
        self.assertEqual([len(b) for b in got], [4, 4, 2, 4, 4, 2])
        self.assertEqual(sorted(i for b in got[:3] for i in b), list(range(10)))
        self.assertNotEqual(got[:3], got[3:])

    def test_mean_loss_matches_per_document_loss(self):
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8)).eval()
        docs = [[3, 4, 5, 6], [7, 8], [9, 10, 11]]
        got = sdf.mean_loss(model, docs, pad_id=0, batch_size=2, device="cpu")
        total = sum(model(torch.tensor([d]), labels=torch.tensor([d])).loss.item() * (len(d) - 1)
                    for d in docs)
        self.assertAlmostEqual(got, total / sum(len(d) - 1 for d in docs), places=5)

    def test_target_regex_selects_the_seven_decoder_modules(self):
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8))
        hits = [n for n, _ in model.named_modules() if re.fullmatch(sdf.TARGET_REGEX, n)]
        self.assertEqual(len(hits), 2 * 7)
        self.assertEqual({n.rsplit(".", 1)[1] for n in hits},
                         {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
        # Qwen3.6's linear-attention projections stay unadapted.
        for name in ["model.language_model.layers.0.linear_attn.out_proj",
                     "model.language_model.layers.0.linear_attn.in_proj_qkv"]:
            self.assertIsNone(re.fullmatch(sdf.TARGET_REGEX, name))
        self.assertIsNotNone(re.fullmatch(sdf.TARGET_REGEX, "model.language_model.layers.3.self_attn.o_proj"))


push_spec = importlib.util.spec_from_file_location(
    "push", Path(__file__).parents[1] / "sdf/push.py")
push_sdf = importlib.util.module_from_spec(push_spec)
push_spec.loader.exec_module(push_sdf)


class FakeHub:
    """Stores uploaded files in memory and reports them back the way get_paths_info does."""
    def __init__(self, corrupt=None):
        self.files, self.corrupt, self.uploads = {}, corrupt, 0

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, commit_message):
        self.uploads += 1
        for name in allow_patterns:
            data = (Path(folder_path) / name).read_bytes()
            self.files[f"{path_in_repo}/{name}"] = data + (b"x" if name == self.corrupt else b"")
        return types.SimpleNamespace(oid="c0ffee" * 6)

    def get_paths_info(self, repo, paths, expand, repo_type):
        out = []
        for p in paths:
            if p in self.files:
                data = self.files[p]
                lfs = types.SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if p.endswith(".safetensors") else None
                out.append(types.SimpleNamespace(path=p, size=len(data), lfs=lfs))
        return out


class PushTests(unittest.TestCase):
    def make_fact(self, root):
        d = Path(root) / "cake_bake"
        d.mkdir()
        for name in push_sdf.FILES:
            (d / name).write_bytes(name.encode() * 3)
        (d / "README.md").write_text("PEFT card")
        return d

    def test_push_verifies_marks_and_skips(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.make_fact(root)
            hub = FakeHub()
            self.assertIn("uploaded and verified", push_sdf.push(hub, "u/r", d))
            self.assertNotIn("cake_bake/README.md", hub.files)
            self.assertTrue((d / "uploaded.json").exists())
            self.assertIn("already uploaded", push_sdf.push(hub, "u/r", d))
            self.assertEqual(hub.uploads, 1)
            (d / "adapter_model.safetensors").write_bytes(b"retrained")
            push_sdf.push(hub, "u/r", d)
            self.assertEqual(hub.uploads, 2)

    def test_push_refuses_a_mismatched_upload(self):
        with tempfile.TemporaryDirectory() as root:
            d = self.make_fact(root)
            with self.assertRaises(RuntimeError):
                push_sdf.push(FakeHub(corrupt="adapter_model.safetensors"), "u/r", d)
            self.assertFalse((d / "uploaded.json").exists())


if __name__ == "__main__":
    unittest.main()
