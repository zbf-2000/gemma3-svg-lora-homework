# Gemma 3 270M 的 SVG Logo 生成实验

## 项目简介

这个项目使用 LoRA 微调 Gemma 3 270M，使模型根据文字描述生成 SVG Logo。实验主要包含三个部分：

1. 编写 `reward.py`，自动检查 SVG 的格式、几何、安全性、配色和提示词特征；
2. 使用训练集对 Gemma 3 270M 进行 LoRA 微调；
3. 在相同的 17 条验证提示上比较基础模型和 LoRA 模型。

原始 SVG 通常包含很长的 path 和较复杂的图层。实际训练时，小模型很容易重复生成或在 SVG 闭合前达到长度上限。因此我将训练目标简化为短小、完整且配色相关的 SVG，让模型优先学习正确的 SVG 文档结构和基本图形。

## 项目结构

```text
adapter/                       LoRA 权重
data/                          规则化后的训练和验证数据
logo-detailed-prompt-main/     原始数据集
tests/                         reward 和训练损失测试
prepare_compact_data.py        生成 compact 数据
train.py                       LoRA 训练脚本
evaluate.py                    Base 与 LoRA 对比评测
reward.py                      SVG 自动评分
train_config.yaml              训练和生成配置
training_metrics.json          训练指标
results.json                   完整验证集结果
report.md                      实验报告
```

本地的 `models/` 用于保存基座模型，`runs/` 用于保存训练 checkpoint。这两个目录没有放入仓库。

## 环境

实验环境为 Python 3.11，主要依赖 PyTorch、Transformers、PEFT 和 Accelerate。

使用 Conda 创建环境：

```powershell
conda env create -f environment.yml
conda activate homework
```

也可以使用 requirements 文件安装：

```powershell
conda create -n homework python=3.11 pip -y
conda activate homework
python -m pip install -r requirements.txt
```

下载基座模型：

```powershell
modelscope download --model google/gemma-3-270m-it --local_dir models/gemma-3-270m-it
```

## 数据处理

原始训练集有 219 条记录，其中两条 prompt 为 `placeholder`，清理后剩余 217 条。

`prepare_compact_data.py` 保留原始 prompt，并将较长的参考 SVG 转换为简短的两图元 SVG。生成的目标统一使用正确的 namespace 和 `viewBox="0 0 256 256"`，颜色优先从 prompt 和原始参考 SVG 中提取。

重新生成 compact 数据：

```powershell
python prepare_compact_data.py
```

生成文件为：

```text
data/train_compact.jsonl
data/valid_compact.jsonl
data/compact_manifest.json
```

## 训练

运行训练：

```powershell
python train.py --config train_config.yaml
```

主要训练参数：

| 参数 | 设置 |
|---|---:|
| 训练样本 | 217 |
| Epoch | 3 |
| Batch size | 1 |
| 梯度累积 | 8 |
| 学习率 | 1e-4 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| 最大序列长度 | 768 |

训练只对 assistant 部分计算 loss。由于模型词表较大，训练脚本使用分块交叉熵以降低 6GB 显存上的峰值占用。

本次训练耗时约 4 分 52 秒，结果为：

```text
train_loss = 0.43845
eval_loss  = 0.28424
```

LoRA 权重保存在 `adapter/`，训练过程中的 checkpoint 保存在 `runs/train/`。

## 评测

运行完整验证集评测：

```powershell
python -u evaluate.py --config train_config.yaml --output results.json --restart
```

基础模型和 LoRA 模型使用相同的 17 条 prompt 和相同的生成参数：

```text
greedy decoding
max_new_tokens = 256
repetition_penalty = 1.0
no_repeat_ngram_size = 0
```

生成在 `</svg>`、`<eos>` 或 `<end_of_turn>` 处停止。评测结果逐条保存在临时文件中，因此中断后可以继续运行：

```powershell
python -u evaluate.py --config train_config.yaml --output results.json
```

## 实验结果

| 指标 | Base | LoRA | 变化 |
|---|---:|---:|---:|
| 平均 reward | 9.118 | 93.737 | +84.619 |
| 有效 SVG 比例 | 0.00000 | 1.00000 | +1.00000 |
| 完整 SVG 比例 | 0.35294 | 1.00000 | +0.64706 |
| 达到长度上限比例 | 0.41176 | 0.00000 | -0.41176 |

LoRA 模型在 17 条验证数据上都生成了完整、可解析并且 viewBox 正确的 SVG。基础模型则经常出现 namespace 错误、重复生成和输出截断。

## 局限

这个实验主要改善了 SVG 的合法性、闭合稳定性和配色。由于 compact 训练目标过于简单，验证集上的 LoRA 输出都采用了两个同心圆的结构，只根据 prompt 改变颜色。

因此，较高的 reward 并不表示模型已经能够理解复杂 Logo 的构图。房屋、人物、盾牌、手掌和渐变等高级视觉元素仍然没有被可靠生成。更详细的样例和 Goodhart 效应分析记录在 `report.md` 中。

## 测试

运行测试：

```powershell
python -B -m unittest discover -s tests -v
python -m py_compile reward.py train.py evaluate.py prepare_compact_data.py
```

当前 14 个测试均通过。
