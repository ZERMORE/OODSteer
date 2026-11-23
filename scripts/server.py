"""Single server entry point. Called inside conda env by run.sh."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oodsteer.paths import configure_storage


def command(args, log=None):
    print("+", " ".join(map(str,args)), flush=True)
    if log is None:
        subprocess.run(args, cwd=ROOT, check=True)
        return
    with Path(log).open("x", encoding="utf-8") as f:
        with subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)


def select_tasks(action, requested, strategyqa):
    if requested:
        tasks = list(dict.fromkeys(requested))
    elif action == "smoke":
        tasks = ["gsm8k"]
    else:
        tasks = ["csqa", "gsm8k", "aqua"]
        if strategyqa.is_file():
            tasks.append("strategyqa")
        else:
            print(f"StrategyQA skipped: missing {strategyqa}", flush=True)
    if "strategyqa" in tasks and not strategyqa.is_file():
        raise FileNotFoundError(f"StrategyQA requested but missing: {strategyqa}")
    return tasks


def experiment_command(a, task, method, output, dtype):
    cmd = [sys.executable, "-u", "-m", "oodsteer.cli", "--model", a.model,
           "--preset", a.preset, "--task", task, "--method", method,
           "--prompt-file", str(ROOT / "prompts" / f"{task}.txt"),
           "--dtype", dtype, "--max-new-tokens", str(a.max_new_tokens),
           "--output", str(output)]
    cmd += ["--stop-policy", getattr(a,"stop_policy","answer"), "--scoring", getattr(a,"scoring","compatible-v2")]
    if a.action == "smoke":
        cmd += ["--limit", "3"]
    if a.reuse_recipe:
        cmd += ["--reuse-recipe", a.reuse_recipe]
    return cmd


def launch_background(a, paths, argv):
    if a.action not in ("smoke", "main"):
        raise ValueError("--background supports smoke/main only")
    name = a.run_name or f"{a.action}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    directory=paths["logs"] / "background"
    directory.mkdir(parents=True,exist_ok=True)
    log=directory/f"{name}.log"
    pid_file=directory/f"{name}.pid"
    status=directory/f"{name}.status.json"
    if any(path.exists() for path in (log,pid_file,status)):
        raise FileExistsError(f"Background run name already used: {name}")
    args=[arg for arg in argv if arg != "--background"]
    if a.run_name is None:
        args += ["--run-name",name]
    env=dict(os.environ, OODSTEER_BACKGROUND_STATUS=str(status))
    with log.open("x",encoding="utf-8") as output:
        process=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),*args],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,
            stderr=subprocess.STDOUT,start_new_session=True)
    pid_file.write_text(str(process.pid)+"\n",encoding="utf-8")
    # Child owns status updates. Its PID also identifies its new process group.
    print(f"Submitted: {name}\nPID / PGID: {process.pid}\nLog: {log}\nStatus: {status}\nPID file: {pid_file}",flush=True)
    print(f"Follow: tail -f {log}",flush=True)
    print("Submission succeeded; inspect log/status to confirm checks and experiment startup.",flush=True)


def write_status(state, error=None):
    location=os.environ.get("OODSTEER_BACKGROUND_STATUS")
    if not location:
        return
    value=dict(state=state,pid=os.getpid(),updated=datetime.now().isoformat(),error=error)
    path=Path(location)
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "smoke", "main"])
    p.add_argument("--background", action="store_true", help="Detach main/smoke from SSH; save combined log and PID")
    p.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES; otherwise inherit allocation")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--preset", default="qwen3b", choices=["llama1b","qwen3b","qwen14b"])
    p.add_argument("--tasks", nargs="+", choices=["csqa","gsm8k","aqua","strategyqa"])
    p.add_argument("--run-name")
    p.add_argument("--dtype", default="auto", choices=["auto","bfloat16","float16","float32"])
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--reuse-recipe")
    p.add_argument("--stop-policy",choices=["answer","question","eos"],default="answer")
    p.add_argument("--scoring",choices=["compatible-v2","legacy-v1"],default="compatible-v2")
    a = p.parse_args()
    if a.run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.run_name):
        p.error("run-name must be a simple directory name")
    if a.max_new_tokens < 1:
        p.error("max-new-tokens must be positive")
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    paths = configure_storage()
    if a.background:
        launch_background(a, paths, sys.argv[1:])
        return
    write_status("running")
    print("GPU visibility:", os.environ.get("CUDA_VISIBLE_DEVICES", "inherited/all visible"), flush=True)
    print("Data root:", paths["root"], "HF_HOME:", os.environ["HF_HOME"], flush=True)
    # Check every required dependency and the actual GPU before experiment output.
    command([sys.executable,"-m","pip","check"])
    command([sys.executable,str(ROOT/"scripts/check_env.py")])
    command([sys.executable,"-m","unittest","discover","-s","tests","-v"])
    if a.action == "check":
        print("All checks passed", flush=True)
        return
    import torch
    dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if a.dtype == "auto" else a.dtype
    tasks = select_tasks(a.action,a.tasks,paths["strategyqa"])
    run_name = a.run_name or f"{a.action}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    out = paths["results"] / run_name / a.preset
    logs = paths["logs"] / run_name / a.preset
    out.mkdir(parents=True,exist_ok=False)
    logs.mkdir(parents=True,exist_ok=False)
    (logs/"run.json").write_text(json.dumps(dict(vars(a), tasks_resolved=tasks,
        dtype_resolved=dtype, output=str(out)),indent=2),encoding="utf-8")
    with (logs/"environment-resolved.txt").open("x",encoding="utf-8") as f:
        subprocess.run([sys.executable,"-m","pip","freeze"],stdout=f,check=True)
    print(f"Tasks: {tasks}; dtype: {dtype}; results: {out}; logs: {logs}",flush=True)
    results=[]
    for task in tasks:
        for method in ("standard","usteer"):
            result=out/f"{task}-{method}.jsonl"
            command(experiment_command(a,task,method,result,dtype),logs/f"{task}-{method}.log")
            results.append(result)
    from oodsteer.report import summarize
    summary=[summarize(path) for path in results]
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Complete. Results: {out}; logs: {logs}",flush=True)

if __name__ == "__main__":
    try:
        main()
        write_status("complete")
    except subprocess.CalledProcessError as exc:
        write_status("failed", str(exc))
        print(f"Command failed (exit {exc.returncode}); stopped. Existing outputs preserved.",file=sys.stderr)
        sys.exit(exc.returncode if exc.returncode>0 else 1)

    except KeyboardInterrupt:
        write_status("interrupted")
        sys.exit(130)
    except Exception as exc:
        write_status("failed", str(exc))
        raise
