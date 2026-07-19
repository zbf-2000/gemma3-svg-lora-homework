# Part B：Gemma 3 270M 生成 SVG Logo

本项目已经形成可提交的最终结果：在原始 `valid.jsonl` 全部 17 条提示上，基座模型平均 reward 为 `9.118`、有效 SVG 比例为 `0%`；LoRA adapter 平均 reward 为 `93.737`、有效 SVG 比例为 `100%`。

提升主要来自 SVG 闭合、正确 namespace/viewBox、基础几何和配色稳定性。LoRA 输出高度模板化，17 条结果均为两个同心圆，只改变颜色；详细边界与 Goodhart 分析见 `report.md`。

## 提交目录

```text
adapter/
  adapter_config.json
  adapter_model.safetensors
data/
  compact_manifest.json
  train_compact.jsonl
  valid_compact.jsonl
logo-detailed-prompt-main/
  train.jsonl
  valid.jsonl
tests/
  test_reward.py
  test_train_loss.py
prepare_compact_data.py
train.py
evaluate.py
reward.py
train_config.yaml
training_metrics.json
results.json
report.md
README.md
requirements.txt
environment.yml
```

`models/` 和 `runs/` 是本地模型与 checkpoint，已通过 `.gitignore` 排除，不需要提交。

## 1. 配置环境

已有环境：

```powershell
conda activate homework
```

在新机器上创建同名环境：

```powershell
conda env create -f environment.yml
conda activate homework
```

也可手动安装：

```powershell
conda create -n homework python=3.11 pip -y
conda activate homework
python -m pip install -r requirements.txt
```

检查 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## 2. 下载基座模型

```powershell
modelscope download --model google/gemma-3-270m-it --local_dir models/gemma-3-270m-it
```

## 3. 生成 compact curriculum

最终训练保留原始 217 条有效 prompt，但把长 Sonnet SVG 规则化为短小、合法、配色相关的两图元 SVG。生成过程完全确定，执行：

```powershell
python prepare_compact_data.py
```

该命令生成：

```text
data/train_compact.jsonl
data/valid_compact.jsonl
data/compact_manifest.json
```

仓库已经包含这三个最终文件；只有需要从原始数据重新构造时才需再次运行。

## 4. 运行检查

```powershell
python -B -m unittest discover -s tests -v
python -m py_compile reward.py train.py evaluate.py prepare_compact_data.py
```

当前共 14 个测试，全部通过。

## 5. 重新训练（可选）

最终 adapter 已包含在 `adapter/`，提交作业不需要再次训练。若要从头复现：

```powershell
python train.py --config train_config.yaml
```

最终配置为：

- 217 条 compact 训练样本、17 条 compact validation 样本；
- `train_max_length=768`，无截断、无超长样本丢弃；
- assistant-only loss；
- LoRA rank 8、alpha 16、dropout 0.05；
- batch size 1、梯度累积 8、学习率 `1e-4`；
- 3 epochs、84 optimizer steps；
- checkpoint 写入 `runs/train/`，最终权重写入 `adapter/`。

本次训练耗时约 291.8 秒，结果为：

```text
train_loss = 0.43845
eval_loss  = 0.28424
```

检查训练产物：

```powershell
Get-Item adapter\adapter_config.json
Get-Item adapter\adapter_model.safetensors
Get-Content training_metrics.json -Encoding UTF8
```

## 6. 正式评测

从头运行完整 17 条评测：

```powershell
python -u evaluate.py --config train_config.yaml --output results.json --restart
```

固定生成设置为 greedy、`max_new_tokens=256`、`repetition_penalty=1.0`、`no_repeat_ngram_size=0`；基座和 LoRA 完全一致。停止条件为 `</svg>`、`<eos>` 和 `<end_of_turn>`。

评测每完成一条就原子写入 `results.json.partial`。若意外中断，使用以下命令续跑，不要加 `--restart`：

```powershell
python -u evaluate.py --config train_config.yaml --output results.json
```

脚本会校验配置、reward、评测代码、adapter 和验证集哈希，只恢复签名一致的任务。完成后生成 `results.json` 并删除 partial。

快速 pilot 仅用于调试，不能作为最终结果：

```powershell
python -u evaluate.py --config train_config.yaml --output pilot.json --max-samples 3 --restart
```

## 7. 最终结果

| 指标 | Base | LoRA | Δ |
|---|---:|---:|---:|
| 平均 reward | 9.118 | 93.737 | +84.619 |
| 有效 SVG 比例 | 0.00000 | 1.00000 | +1.00000 |
| 完整 SVG 比例 | 0.35294 | 1.00000 | +0.64706 |
| 长度上限比例 | 0.41176 | 0.00000 | -0.41176 |

完整摘要和全部逐样本输出位于 `results.json`，实验过程、三组真实样例和局限分析位于 `report.md`。

## 8. 最低提交文件

题目要求至少提交：

```text
adapter/adapter_config.json
adapter/adapter_model.safetensors
reward.py
train_config.yaml
results.json
report.md
```

建议同时提交本 README、训练/评测/数据生成脚本、compact 数据、依赖文件和测试，以便完整复现。
