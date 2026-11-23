import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import os
from unittest.mock import patch

spec=importlib.util.spec_from_file_location("oodsteer_server",Path(__file__).resolve().parents[1]/"scripts/server.py")
server=importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

class LauncherTests(unittest.TestCase):
    def test_task_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"strategyqa.jsonl"
            self.assertEqual(server.select_tasks("smoke",None,path),["gsm8k"])
            self.assertEqual(server.select_tasks("main",None,path),["csqa","gsm8k","aqua"])
            with self.assertRaises(FileNotFoundError):
                server.select_tasks("main",["strategyqa"],path)
            path.touch()
            self.assertIn("strategyqa",server.select_tasks("main",None,path))

    def test_experiment_command(self):
        args=argparse.Namespace(action="smoke",model="local-model",preset="qwen3b",max_new_tokens=512,reuse_recipe="recipe.json")
        command=server.experiment_command(args,"gsm8k","usteer",Path("out.jsonl"),"float16")
        self.assertEqual(command[command.index("--limit")+1],"3")
        self.assertEqual(command[command.index("--dtype")+1],"float16")
        self.assertEqual(command[command.index("--reuse-recipe")+1],"recipe.json")
        args.action="main"
        self.assertNotIn("--limit",server.experiment_command(args,"gsm8k","standard",Path("out.jsonl"),"bfloat16"))

    def test_failed_command_preserves_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/"failure.log"
            with self.assertRaises(subprocess.CalledProcessError) as ctx:
                server.command([sys.executable,"-u","-c","print('diagnostic'); raise SystemExit(7)"],log)
            self.assertEqual(ctx.exception.returncode,7)
            self.assertIn("diagnostic",log.read_text())

    def test_background_launch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES="3"):
            paths={"logs":Path(tmp)}
            args=argparse.Namespace(action="main",run_name="test-gpu3")
            with patch.object(server.subprocess,"Popen") as popen:
                popen.return_value.pid=12345
                server.launch_background(args,paths,["main","--gpu","3","--background","--run-name","test-gpu3"])
                positional,kw=popen.call_args
                self.assertNotIn("--background",positional[0])
                self.assertTrue(kw["start_new_session"])
                self.assertEqual(kw["stdin"],subprocess.DEVNULL)
                self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"],"3")
                self.assertEqual((Path(tmp)/"background/test-gpu3.pid").read_text().strip(),"12345")
                with self.assertRaises(FileExistsError):
                    server.launch_background(args,paths,["main","--background"])

    def test_status_record(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"status.json"
            with patch.dict(os.environ,OODSTEER_BACKGROUND_STATUS=str(path)):
                server.write_status("failed","test failure")
            result=json.loads(path.read_text())
            self.assertEqual(result["state"],"failed")
            self.assertEqual(result["error"],"test failure")
