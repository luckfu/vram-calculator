# VRAM Calculator

VRAM Calculator 是一个命令行工具，用于估算运行大型语言模型（LLM）所需的 GPU 显存（VRAM）。它可以帮助研究人员、开发者和 AI 爱好者在部署或微调模型前评估硬件需求。

## 功能特点

- **精确估算** - 计算运行 LLM 所需的 VRAM，考虑模型权重、KV 缓存、激活项和系统开销
- **预定义模型库** - 内置多种流行模型（如 Llama 3.2、Gemma 2、Mistral、Phi 3/4 等）
- **自定义参数** - 支持自定义模型大小、上下文长度和其他参数
- **多种精度支持** - 计算不同数据类型（float32、float16、bfloat16、int8、int4）的显存需求
- **优化技术** - 考虑 PagedAttention 和 Tensor Parallelism 等优化方法的影响
- **GPU 推荐** - 根据计算结果推荐适合的 GPU 型号
- **交互式界面** - 提供友好的命令行交互体验

## 安装

克隆仓库并确保安装了 Python 3.6 或更高版本：

```bash
git clone https://github.com/yourusername/vram-calculator.git
cd vram-calculator
```

本工具没有外部依赖，使用 Python 标准库即可运行。

## 使用方法

### 交互模式

直接运行脚本，无需任何参数，进入交互式问答模式：

```bash
base ❯ python vram_calculator_cli.py   

--- VRAM 计算器 (交互模式) ---

可用的预定义模型:
  1. Llama 3.2 (1B) (1B 参数, 2048 上下文长度)
  2. Moondream 2 (1.4B 参数, 2048 上下文长度)
  3. Gemma 2 (2B) (2B 参数, 2048 上下文长度)
  4. Llama 3.2 (3B) (3B 参数, 2048 上下文长度)
  5. Phi 3 Mini (3.8B 参数, 2048 上下文长度)
  6. Code Llama (7B 参数, 4096 上下文长度)
  7. DeepSeek-R1 (7B 参数, 4096 上下文长度)
  8. LLaVA (7B 参数, 4096 上下文长度)
  9. Llama 2 Uncensored (7B 参数, 4096 上下文长度)
  10. Mistral (7B 参数, 8192 上下文长度)
  11. Neural Chat (7B 参数, 4096 上下文长度)
  12. Starling (7B 参数, 4096 上下文长度)
  13. Llama 3.1 (8B 参数, 4096 上下文长度)
  14. Gemma 2 (9B) (9B 参数, 4096 上下文长度)
  15. Solar (10.7B 参数, 4096 上下文长度)
  16. Llama 3.2 Vision (11B 参数, 4096 上下文长度)
  17. Phi 4 (14B 参数, 2048 上下文长度)
  18. Gemma 2 (27B) (27B 参数, 4096 上下文长度)
  19. Llama 3.3 (70B 参数, 4096 上下文长度)
  20. Llama 3.2 Vision (90B) (90B 参数, 8192 上下文长度)
  21. Llama 3.1 (405B) (405B 参数, 4096 上下文长度)
  22. DeepSeek-R1 (671B) (671B 参数, 4096 上下文长度)
  0. 自定义模型参数
请按编号选择模型 (输入 0 进行自定义): 21

已选择预定义模型: Llama 3.1 (405B) (405B 参数, 默认上下文长度: 4096)
是否使用默认上下文长度 (4096)? (Y/n): y

可用的数据精度类型 (Data Type):
  1. float32
  2. float16
  3. bfloat16
  4. int8
  5. int4
请按编号选择数据精度类型 (默认为 2. float16): 5
请输入 Batch Size (默认为 1): 32
是否使用 PagedAttention 优化? (y/N): y
是否使用 Tensor Parallelism? (y/N): y
请输入 GPU 数量 (用于 Tensor Parallelism, 默认为 1): 32
是否为训练模式 (计算包含 activations 的显存)? (y/N): n

预估 VRAM: 690.11 GB
显存占用分布明细:
  - 模型权重 (Model Weights): 5.89 GB
  - KV Cache: 621.48 GB
  - 激活项 (Activations): 0.00 GB
  - 额外开销 (Overhead): 62.74 GB
```

按照提示选择预定义模型或输入自定义参数。

### 命令行模式

使用命令行参数直接计算：

```bash
python vram_calculator_cli.py --model-name "Llama 3.1" --data-type float16 --batch-size 4 --context-length 8192
```

#### 可用参数

- `--model-name` - 选择预定义模型（例如 "Llama 3.1"、"Mistral"）
- `--model-size` - 模型参数大小（单位：十亿，例如 7 表示 7B）
- `--data-type` - 数据精度类型（float32、float16、bfloat16、int8、int4）
- `--context-length` - 上下文长度（tokens）
- `--batch-size` - 批处理大小
- `--paged-attention` - 启用 PagedAttention 优化
- `--tensor-parallelism` - 启用 Tensor Parallelism
- `--num-gpus` - GPU 数量（用于 Tensor Parallelism）
- `--training` - 计算训练模式所需的 VRAM（包含激活项）

## 计算示例

以下是一个计算示例的输出：

```
预估 VRAM: 14.32 GB

显存占用分布明细:
  - 模型权重 (Model Weights): 7.00 GB
  - KV Cache (KV Cache): 6.00 GB
  - 激活项 (Activations): 0.00 GB
  - 额外开销 (Overhead): 1.32 GB

推荐 GPU (基于可用 VRAM):
  - NVIDIA RTX 4080 (16 GB)
  - NVIDIA RTX 4090 (24 GB)
  - NVIDIA RTX 5090 (32 GB)
  - NVIDIA A100 (40 GB)
  - NVIDIA L40 (48 GB)
  - NVIDIA A100 80GB (80 GB)
  - NVIDIA H100 (80 GB)
  - AMD MI250X (128 GB)
  - AMD MI300X (192 GB)
```

## 技术细节

VRAM 计算考虑以下因素：

1. **模型权重** - 基于参数数量和数据类型
2. **KV 缓存** - 基于上下文长度、批处理大小和嵌入维度
3. **激活项** - 仅在训练模式下计算
4. **系统开销** - 估算为以上总和的 10%

优化技术：
- **PagedAttention** - 可减少约 50% 的 KV 缓存显存
- **Tensor Parallelism** - 在多个 GPU 之间分配模型权重

## 支持的模型

工具内置多种流行的大型语言模型，包括但不限于：

- Llama 3.1/3.2/3.3 系列
- Gemma 2 系列 (2B, 9B, 27B)
- Mistral 7B
- Phi 3 Mini 和 Phi 4
- Code Llama
- LLaVA
- DeepSeek-R1
- Solar
- 以及更多...

## 支持的 GPU

工具包含多种 NVIDIA 和 AMD GPU 的显存信息，包括：

- 消费级 GPU（RTX 3060 到 RTX 5090）
- 专业级 GPU（A100、H100、L40）
- AMD 数据中心 GPU（MI250X、MI300X）
- Google TPU

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进这个工具。您可以：

- 添加新的模型到 `data/models-and-gpus.json`
- 添加新的 GPU 型号和显存信息
- 改进 VRAM 计算算法
- 增强用户界面体验

## 许可证

[MIT License](LICENSE)