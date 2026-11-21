import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from oodsteer.paths import configure_storage, default_recipe, resolve_model
from oodsteer.data import load

class StorageTests(unittest.TestCase):
    def test_recipe_and_legacy_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            base = Path(tmp)
            configure_storage(base)
            legacy = base / "huggingface/datasets"
            legacy.mkdir(parents=True)
            recipe = base / "results/OODmethod/qwen25-paper-v4-matched/recipe.json"
            recipe.parent.mkdir(parents=True)
            recipe.write_text(json.dumps({"datasets":{"gsm8k":{
                "repo_id":"openai/gsm8k", "config":"main", "split":"test", "revision":"fixed-sha"}}}))
            self.assertEqual(default_recipe(), recipe)
            calls=[]
            def fake(repo, **kw):
                calls.append(kw)
                self.assertEqual(kw["revision"], "fixed-sha")
                self.assertIn("download_config", kw)
                if Path(kw["cache_dir"]) != legacy:
                    raise FileNotFoundError("not cached")
                return [{"question":"one?", "answer":"#### 1"}]
            module=types.SimpleNamespace(load_dataset=fake, DownloadConfig=lambda **kw:kw)
            with patch.dict(sys.modules, datasets=module):
                metadata={}
                rows=load("gsm8k",allow_partial=True,metadata=metadata)
            self.assertEqual(rows[0]["answer"], "1")
            self.assertEqual(len(calls), 2)
            self.assertEqual(metadata["cache_dir"], str(legacy))
            self.assertEqual(metadata["revision"], "fixed-sha")

    def test_model_reuse_requires_complete_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"snapshot"
            folder.mkdir()
            config=folder/"config.json"
            config.write_text("{}")
            hub=types.SimpleNamespace(try_to_load_from_cache=lambda *a,**k:str(config))
            with patch.dict(sys.modules, huggingface_hub=hub):
                self.assertEqual(resolve_model("Qwen/Qwen2.5-3B-Instruct",tmp),"Qwen/Qwen2.5-3B-Instruct")
                (folder/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"a":"part.safetensors"}}))
                self.assertEqual(resolve_model("Qwen/Qwen2.5-3B-Instruct",tmp),"Qwen/Qwen2.5-3B-Instruct")
                (folder/"part.safetensors").touch()
                self.assertEqual(resolve_model("Qwen/Qwen2.5-3B-Instruct",tmp),str(folder))

if __name__ == "__main__":
    unittest.main()
