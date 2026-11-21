# 构建验证（2026-11-21）

仅构建代码，没有下载模型权重、训练或运行真实 benchmark。

本地 Python 3.12 / PyTorch 2.3.0+cu121 下，`python -m unittest discover -s tests -v` 的 4 个测试通过；`python -m compileall -q oodsteer scripts tests` 通过。

检查包含 NCI+RMSNorm 解析梯度对 autograd、层索引、hook 单位步长/仅末 token/清理、答案提取。梯度检查使用两种 epsilon，避免仅验证忽略 epsilon 的近似式。

本机没有 transformers，尚未验证真实 Hugging Face decoder/generate 接口和数据集在线加载。服务器安装文档版本后必须执行 README 的环境检查与 smoke 命令，再跑完整实验。PyTorch 2.5.1 / transformers 4.46.3 是选定服务器基线版本，尚未在本机共同安装验证。

已检查 canonical prompt 个数：GSM8K 8、CSQA 7、StrategyQA 6、AQuA 4。Prompt 格式和来源区别见 README。

## 服务器共享路径更新

默认根目录 `/home/wxh/wdl/data` 已写入 paths.py。模型缓存放 models/huggingface，数据集处理缓存放 datasets/huggingface，StrategyQA 默认文件放 datasets/strategyqa。结果和批量日志放 results/OODSteer 与 logs/OODSteer。通过临时目录检查缓存环境变量配置、数据集 cache_dir 传递和 StrategyQA 默认读取；未访问服务器文件系统，也未迁移历史数据。
