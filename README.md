# OODSteer：USteer 独立实现

论文：[Alleviating Hallucination in Reasoning Tasks with Training-Free Uncertainty-Guided Steering](https://arxiv.org/html/2609.38962v2)，本项目依据 v2（2026-11-09）构建。

截至本次核查（2026-11-21），未找到可确认的官方公开代码。论文 checklist 的 Open access to data and code 明确回答 No，说明 release 尚需 legal/compliance review。本目录是独立实现，尚未运行真实模型实验，不声称已复现论文表格。

## 实现范围与依据

- `oodsteer/steering.py` 实现 Eq. 2、3、6–8 和 Algorithm 1：逐层 logit lens，选当前层 argmax token 的 unembedding weight，解析 NCI 梯度，经最终 RMSNorm Jacobian 回到原 hidden state，再加 alpha 倍单位梯度。不训练、不改变权重、不需要 backward。
- Eq. 2 **没有除以 unembedding weight 的范数**，本实现保留该定义。数值运算用 FP32，词表投影用模型 dtype。
- RMSNorm 使用真实 epsilon；原文 Eq. 7 省略 epsilon 和一个正的整体因子。后者会在单位梯度归一化中抵消。
- 支持标准 greedy、NCI USteer、温度采样、alpha/depth/显式层号消融；保存逐题生成、精确匹配、解析失败、截断、耗时和 bootstrap 95% CI。
- 主实验入口覆盖 Llama-3.2-1B-Instruct、Qwen2.5-3B-Instruct；另有 Qwen2.5-14B 参数。DoLa、ITI、TruthX、fDBD、softmax 引导和 Qwen3-32B 附录实验尚未实现。

## 关键设置与不确定细节

论文 §4.1 设置：CSQA validation 1221；StrategyQA dev 229；GSM8K test 1319；AQuA validation 254。程序默认检查完整集大小。StrategyQA 的 229 条 dev 的确切 ID/划分没有在已核查原文中提供，本项目默认读取 `/home/wxh/wdl/data/datasets/strategyqa/strategyqa-dev.jsonl`，需提供已确认来源的 dev 文件，不能用任意 229 条或随机切分替代并声称精确复现。

| 模型 preset | 深度 | CSQA alpha | StrategyQA | GSM8K | AQuA |
|---|---:|---:|---:|---:|---:|
| llama1b | 3 | .2 | .1 | .3 | .03 |
| qwen3b | 8 | .9 | .3 | .5 | .3 |
| qwen14b | 3 | .7 | 1.1 | 1.1 | 1.5 |

层号为零起始 decoder block **输出**，排除最后一个 block；16 层 Llama 使用 12/13/14，36 层 Qwen 使用 27–34。原文称 last d intermediate layers，但没有官方实现可消除层索引歧义，可用 `--layers` 或 `--depth` 做敏感性检查。

默认仅改变每步最后一个 token 的激活（包括 prefill 的末 token）；`--scope all` 改变 prefill 全部 token，decode 仍是单 token。原文未明确 prefill 范围，所以两者都暴露。使用 KV cache，batch size 1。默认不套 chat template；`--chat` 可作对照。最大输出 512、eager attention、BF16 是本项目工程默认，不是已知作者精确设置。默认 generate 不在答案标记处提前停止，遇 EOS 或长度上限停止；统计截断率并按需要增加上限。若生成了新的 Q:，评分只取第一条回答，耗时仍计实际全部生成。

## Prompt 来源

论文 Appendix A.2 使用 Wei et al. (2022) canonical CoT demonstrations。已附四份 few-shot prefix：GSM8K 8-shot 与 AQuA 4-shot 从 [Wei et al. 原始 TeX](https://arxiv.org/src/2201.11903) 附录提取；CSQA 7-shot、StrategyQA 6-shot 来自 [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) 的 `research/complexity_based_prompting/{csqa,strategyqa}/lib_prompt/prompt_original.txt`。保留相应原始来源文件供核对。它们是 canonical 示例的公开版本，作者使用的确切空格、换行和模型 chat 包装仍未知。遵循原来源使用条款。

评分严格匹配原文 Appendix A.2 的最终标记；缺失标记记错，不从推理过程猜测答案。GSM8K 数字用 Decimal 归一化。Prompt 和数据 SHA256、模型配置、软件版本写入 `.meta.json`，模型权重和 tokenizer 如需严格溯源应使用固定 HF snapshot 路径，并记录下载 revision。

## Linux / Conda 服务器操作

把整个目录上传到服务器，例如 `~/methods/OODSteer`，以下命令在此目录执行。建议先用 Qwen3B（无 Llama 访问授权门槛）。以下采用 PyTorch 2.5.1 CUDA 12.1 wheel；服务器驱动应兼容 CUDA 12.1。系统不必安装同版本 CUDA toolkit。若驱动不兼容，先选匹配驱动的官方 PyTorch wheel。

```bash
cd ~/methods/OODSteer
conda create -n oodsteer python=3.10 -y
conda activate oodsteer
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
python -m pip check
nvidia-smi
python scripts/check_env.py
python -m unittest discover -s tests -v
python -m compileall -q oodsteer scripts tests
python -m pip freeze > environment-resolved.txt
```

推荐单张支持 BF16 的 GPU。1B/3B 可从 16–24 GB 显存尝试（工程估计，随上下文长度变化）；14B 建议 48–80 GB。当前加载为单设备完整权重，不支持量化、跨卡 sharding、vLLM。非 BF16 GPU 可用 `--dtype float16`，记录精度差异。论文 latency 使用 A100 80GB，本实现未针对其低延迟做 kernel 优化，不能假定同等开销。

存储路径已编码到 `oodsteer/paths.py`，不需要手动 export HF_HOME。默认使用以下服务器目录：

```text
/home/wxh/wdl/data/
├── models/huggingface/           # 模型、tokenizer 和 Hub 下载缓存
├── datasets/huggingface/         # HF 数据集 Arrow / 处理缓存
├── datasets/strategyqa/strategyqa-dev.jsonl
├── huggingface/                 # HF 配置和认证等运行文件
├── cache/huggingface-assets/     # HF 辅助缓存
├── results/OODSteer/             # 结果及 meta 文件
└── logs/OODSteer/                # 批量实验日志
```

`--data-root` 优先于 `OODSTEER_DATA_ROOT`，其次为上述默认路径。程序在导入 Hugging Face 库前配置缓存，并显式传入模型/数据集 cache_dir；会覆盖通用 HF 缓存环境变量，以保证项目路径一致。`--data`、`--output` 仍可指定单个文件路径。单次运行不传 `--output` 时，保存到 `results/OODSteer/<preset>/<task>-<method>.jsonl`（位于 data root 下）。批量脚本按 RUN 和 preset 建结果目录并将 stdout/stderr 保存到 logs/OODSteer。已有缓存文件不会自动迁移；若目标目录未命中缓存，将重新下载。数据和缓存目录需有写权限。网络受限时，在联网机器下载模型 snapshot/数据后上传，模型用本地路径，数据用下面的 JSONL 格式。Llama 模型需先获得模型访问权，再用 `HF_HOME=/home/wxh/wdl/data/huggingface huggingface-cli login` 登录（与代码使用同一认证目录）。

先做服务器小样本 smoke（这里才会下载模型和运行推理）：

```bash
python -m oodsteer.cli --model Qwen/Qwen2.5-3B-Instruct --preset qwen3b   --task gsm8k --method standard --prompt-file prompts/gsm8k.txt   --limit 3 --max-new-tokens 128 --output /home/wxh/wdl/data/results/OODSteer/smoke-standard.jsonl
python -m oodsteer.cli --model Qwen/Qwen2.5-3B-Instruct --preset qwen3b   --task gsm8k --method usteer --prompt-file prompts/gsm8k.txt   --limit 3 --max-new-tokens 128 --output /home/wxh/wdl/data/results/OODSteer/smoke-usteer.jsonl
python -m oodsteer.report /home/wxh/wdl/data/results/OODSteer/smoke-*.jsonl
```

完整跑单任务（不依赖 StrategyQA 的本地 dev）：

```bash
for method in standard usteer; do
  python -m oodsteer.cli --model Qwen/Qwen2.5-3B-Instruct --preset qwen3b     --task gsm8k --method "$method" --prompt-file prompts/gsm8k.txt     --output "/home/wxh/wdl/data/results/OODSteer/gsm8k-$method.jsonl"
done
```

准备 StrategyQA：将确认过来源的原始 229 条 dev JSON（list 或 qid->record）上传后：

```bash
python scripts/normalize_strategyqa.py /path/to/strategyqa_dev.json /home/wxh/wdl/data/datasets/strategyqa/strategyqa-dev.jsonl
bash scripts/run_matrix.sh qwen3b Qwen/Qwen2.5-3B-Instruct main-qwen
bash scripts/run_matrix.sh llama1b meta-llama/Llama-3.2-1B-Instruct main-llama
```

所有任务可用 `--data /path/file.jsonl` 替代联网加载，每行包含 `id`、`question`、`answer`，选择题再提供字符串列表 `choices`（包含选项字母）。答案为 A–E / yes,no / 数字字符串。小型自建数据需要 `--allow-partial`；完整实验不要加这个开关。仅行数正确不足以确认划分相同，仍须核对来源与 ID。

消融示例（完整运行中每次使用新 output，避免覆盖）：

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  python -m oodsteer.cli --model Qwen/Qwen2.5-3B-Instruct --preset qwen3b     --task gsm8k --prompt-file prompts/gsm8k.txt --alpha "$alpha"     --output "/home/wxh/wdl/data/results/OODSteer/alpha-$alpha.jsonl"
done
# 采样实验可分别设 --temperature 0.2/0.5/0.8/1.0 与 --seed 1..5。
# 深度与 prefill 范围敏感性：--depth 4；--scope all。
```

以上 alpha sweep 用于复查敏感性；若选新参数，应使用独立 validation/tuning 集，不能把 test 上最好结果当成无偏性能。主实验直接使用论文参数即可。

## 如何读结果

`.jsonl` 保存逐题结果，`.meta.json` 保存配置；report 输出准确率、bootstrap 区间、解析失败数、截断数与 ms/token（排除第一条冷启动耗时，含 prefill，未含 tokenizer）。比较前检查样本 ID、prompt hash、生成上限、精度、scope 一致。先查看错误样本是否因答案标记或截断，再决定是否增加 max-new-tokens；baseline 和 USteer 必须同时修改。

论文 Table 1 的 Standard → USteer：Llama1B CSQA 50.94→54.05、StrategyQA 58.52→63.32、GSM8K 34.19→36.24、AQuA 37.80→39.37；Qwen3B 分别 77.72→79.20、68.56→70.74、79.91→80.97、54.72→55.12。这里仅列原文参考值，本项目没有生成实验结果。原文 Appendix C 说明 greedy 的区间不足以支持确定的显著性结论。

本地验证记录见 `VALIDATION.md`。首次服务器真实运行后，应保留 resolved 环境、模型 snapshot revision、完整日志及结果文件，才能进一步核查与论文数字的差异。
