import argparse
import contextlib
import hashlib
import json
import platform
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF ID or local directory")
    p.add_argument("--preset", choices=["llama1b", "qwen3b", "qwen14b"], required=True)
    p.add_argument("--task", choices=["csqa", "strategyqa", "gsm8k", "aqua"], required=True)
    p.add_argument("--method", choices=["standard", "usteer"], default="usteer")
    p.add_argument("--prompt-file", required=True, help="Few-shot Q/A prefix, UTF-8 text")
    p.add_argument("--data", help="Normalized local JSONL; mandatory for StrategyQA")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--alpha", type=float)
    p.add_argument("--depth", type=int)
    p.add_argument("--layers", help="Explicit zero-based block indices, comma separated")
    p.add_argument("--scope", choices=["last", "all"], default="last")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chat", action="store_true", help="Optional deviation: apply model chat template")
    a = p.parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from .data import load, prompt, extract
    from .steering import USteer, intermediate_layers
    if a.alpha is not None and a.alpha < 0 or a.temperature < 0:
        p.error("alpha and temperature must be nonnegative")
    if a.limit is not None and a.limit <= 0:
        p.error("limit must be positive")
    prefix = Path(a.prompt_file).read_text(encoding="utf-8")
    if not prefix.strip():
        p.error("Empty prompt prefix")
    rows = load(a.task, a.data, a.allow_partial)
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.with_suffix(".meta.json").exists():
        raise FileExistsError(f"Choose a fresh output path: {out}")
    alphas = {"llama1b":[.2,.1,.3,.03], "qwen3b":[.9,.3,.5,.3], "qwen14b":[.7,1.1,1.1,1.5]}
    alpha = a.alpha if a.alpha is not None else alphas[a.preset][["csqa","strategyqa","gsm8k","aqua"].index(a.task)]
    depth = a.depth if a.depth is not None else {"llama1b":3,"qwen3b":8,"qwen14b":3}[a.preset]
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch,a.dtype), attn_implementation="eager").to(a.device).eval()
    expected_layers = {"llama1b":16,"qwen3b":36,"qwen14b":48}[a.preset]
    if model.config.num_hidden_layers != expected_layers:
        raise ValueError("Model layer count does not match preset; use the documented model")
    model.requires_grad_(False)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else intermediate_layers(model.config.num_hidden_layers, depth)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    meta = dict(vars(a), resolved_alpha=alpha, resolved_layers=layers, torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(), model_config=model.config.to_dict(), prompt_sha256=hashlib.sha256(prefix.encode()).hexdigest(), data_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    out.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    manager = USteer(model,layers,alpha,a.scope) if a.method == "usteer" else contextlib.nullcontext()
    def sync():
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
    with manager, torch.no_grad(), out.open("x",encoding="utf-8") as f:
        for i,row in enumerate(rows):
            text = prompt(row,a.task,prefix)
            if a.chat:
                text = tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
            inputs = tok(text,return_tensors="pt").to(a.device)
            if inputs.input_ids.shape[1] + a.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("Prompt plus generation exceeds context window")
            kwargs = dict(max_new_tokens=a.max_new_tokens,do_sample=a.temperature>0,use_cache=True,pad_token_id=tok.pad_token_id)
            if a.temperature>0:
                kwargs.update(temperature=a.temperature,top_p=1.0,top_k=0)
            sync()
            start = time.perf_counter()
            result = model.generate(**inputs,**kwargs)
            sync()
            seconds = time.perf_counter()-start
            tokens = result[0,inputs.input_ids.shape[1]:]
            generated = tok.decode(tokens,skip_special_tokens=True).split("\nQ:")[0]
            answer = extract(generated,a.task)
            record = dict(id=row["id"],gold=row["answer"],prediction=answer,correct=answer==row["answer"],text=generated,tokens=len(tokens),seconds=seconds,truncated=len(tokens)==a.max_new_tokens)
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
            f.flush()
            print(f"{i+1}/{len(rows)} correct={record['correct']} tokens={len(tokens)}",flush=True)
    from .report import summarize
    print(json.dumps(summarize(out),indent=2))

if __name__ == "__main__":
    main()
