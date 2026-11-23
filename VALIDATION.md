# 构建验证（2026-11-21）

仅构建代码，没有下载模型权重、训练或运行真实 benchmark。

本地 Python 3.12 / PyTorch 2.3.0+cu121 下，`python -m unittest discover -s tests -v` 的 4 个测试通过；`python -m compileall -q oodsteer scripts tests` 通过。

检查包含 NCI+RMSNorm 解析梯度对 autograd、层索引、hook 单位步长/仅末 token/清理、答案提取。梯度检查使用两种 epsilon，避免仅验证忽略 epsilon 的近似式。

本机没有 transformers，尚未验证真实 Hugging Face decoder/generate 接口和数据集在线加载。服务器安装文档版本后必须执行 README 的环境检查与 smoke 命令，再跑完整实验。PyTorch 2.5.1 / transformers 4.46.3 是选定服务器基线版本，尚未在本机共同安装验证。

已检查 canonical prompt 个数：GSM8K 8、CSQA 7、StrategyQA 6、AQuA 4。Prompt 格式和来源区别见 README。

## 服务器共享路径更新

默认根目录 `/home/wxh/wdl/data` 已写入 paths.py。模型缓存放 models/huggingface，数据集处理缓存放 datasets/huggingface，StrategyQA 默认文件放 datasets/strategyqa。结果和批量日志放 results/OODSteer 与 logs/OODSteer。通过临时目录检查缓存环境变量配置、数据集 cache_dir 传递和 StrategyQA 默认读取；未访问服务器文件系统，也未迁移历史数据。

## 与旧项目路径对齐

服务器代码目录统一为 /home/wxh/projects/OODSteer；主缓存改为 /home/wxh/wdl/data/cache/huggingface，与 OODmethod 一致。新增旧数据集缓存回退、recipe revision 复用、完整模型 checkpoint 复用测试。服务器真实缓存与网络加载仍需首次 smoke 检查。

## 统一启动入口

新增 run.sh 与 scripts/server.py，支持 setup/check/smoke/main。新增任务选择、实验参数和真实子进程失败日志的 3 项检查，总计 9 项测试。未运行 GPU 模型实验；本机 WSL Bash 不可用，Shell/Conda 入口尚需服务器首次执行验证。

## 评分与生成终止修复

14项测试在本地 transformers 4.46.3 / torch 2.3.0+cu121 环境全部通过。包括随机初始化 tiny Llama 的真实 generate + 强制逐字符输出：确认不会在120时提前截断120.5，会在完整答案后停止；测试无需下载权重。已有主实验四个结果文件已在暂存目录验证重评分：CSQA Standard 77.4775%，USteer 76.6585%；GSM8K Standard 84.2305%，USteer前282题84.7518%。新评分取首个明确答案，不等同于此前audit的最后答案诊断规则。未运行服务器真实模型早停实验。

## 后台运行入口（2026-11-23）

新增 --background，独立session、stdin断开、总日志、PID与状态文件。16项本地测试通过，包括后台启动参数/GPU环境传递、同名日志保护与状态写入；Windows沙箱阻止临时文件原子重命名，获准在沙箱外重跑后全部通过。未在Linux服务器实际执行SSH断连验证。
