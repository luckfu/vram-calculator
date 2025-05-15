import json
import argparse
import math
import sys

# 数据类型大小 (比特)
DATA_TYPE_SIZES = {
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
    "int8": 8,
    "int4": 4,
}

def load_config(config_path="data/models-and-gpus.json"):
    """加载模型和 GPU 配置文件 (JSON格式)。"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：配置文件未在路径 {config_path} 找到。")
        print("请确保 'models-and-gpus.json' 文件位于相对于脚本的 'data' 目录下，或提供正确的路径。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误：无法解析 {config_path} 中的 JSON 数据。请检查文件格式。")
        sys.exit(1)

def calculate_vram(
    model_size_params,  # 单位：十亿
    data_type,
    context_length,
    batch_size,
    use_paged_attention,
    use_tensor_parallelism,
    num_gpus,
    is_training
):
    """计算预估的 VRAM 需求。"""
    param_count = model_size_params * 1_000_000_000  # 将 B (十亿) 单位转为实际参数数量
    data_bits = DATA_TYPE_SIZES[data_type]
    embedding_size = math.floor(4 * math.sqrt(param_count))

    # 模型权重显存
    m_model = (param_count * data_bits) / 8 / (num_gpus if use_tensor_parallelism else 1)

    # KV Cache 显存
    q_kv = 32 if data_type == "float32" else 16  # float32 使用32位，其他使用16位
    m_kv = (batch_size * context_length * embedding_size * 2 * q_kv) / 8
    m_kv_optimized = m_kv * 0.5 if use_paged_attention else m_kv # PagedAttention 可减少约50%

    # Activations 显存 (仅训练时)
    m_activations = (batch_size * context_length * embedding_size * data_bits) / 8 if is_training else 0

    # 开销显存
    m_overhead = (m_model + m_kv_optimized + m_activations) * 0.1  # 10% 开销

    # 总显存 (Bytes)
    total_memory_bytes = m_model + m_kv_optimized + m_activations + m_overhead

    # 转换为 GB
    total_memory_gb = total_memory_bytes / (1024 * 1024 * 1024)

    return {
        "total_gb": round(total_memory_gb, 2),
        "breakdown_gb": {
            "model_weights": round(m_model / (1024 * 1024 * 1024), 2),
            "kv_cache": round(m_kv_optimized / (1024 * 1024 * 1024), 2),
            "activations": round(m_activations / (1024 * 1024 * 1024), 2),
            "overhead": round(m_overhead / (1024 * 1024 * 1024), 2),
        }
    }

def get_user_input(prompt, type_converter, choices=None, default=None):
    """获取并验证用户输入。"""
    while True:
        try:
            user_val_str = input(prompt).strip()
            if not user_val_str and default is not None:
                # 如果允许默认值且用户直接回车
                if isinstance(default, bool): # 特殊处理布尔值默认提示
                    return default
                # 对于其他类型，如果默认值是有效选项，则返回它
                if choices is None or default in choices or type_converter == str or type_converter == float or type_converter == int:
                     return default
            
            user_val = type_converter(user_val_str)
            if choices and user_val not in choices:
                print(f"无效选择。请输入以下选项之一: {', '.join(map(str, choices))}")
                continue
            return user_val
        except ValueError:
            print(f"无效输入。请输入一个有效的 {type_converter.__name__} 类型值。")

def get_bool_input(prompt, default_val_for_prompt=None, default_return_val=False):
    """获取布尔值用户输入 (y/n)。"""
    # 根据default_return_val构造提示中的默认选项
    default_indicator = ''
    if default_val_for_prompt == 'y':
        default_indicator = ' (Y/n)'
    elif default_val_for_prompt == 'n':
        default_indicator = ' (y/N)'
    else: # 无明确倾向的默认或不显示默认倾向
        default_indicator = ' (y/n)'

    full_prompt = f"{prompt}{default_indicator}: "
    
    while True:
        val = input(full_prompt).strip().lower()
        if not val: # 用户直接回车
            return default_return_val
        if val in ['y', 'yes', '是']:
            return True
        if val in ['n', 'no', '否']:
            return False
        print("无效输入。请输入 'y' 或 'n' (或者 '是'/'否')。")


def interactive_mode(config_data):
    """以交互模式运行计算器。"""
    print("\n--- VRAM 计算器 (交互模式) ---")

    model_options = {str(i+1): model['name'] for i, model in enumerate(config_data['commonModels'])}
    print("\n可用的预定义模型:")
    for key, name in model_options.items():
        model_detail = next(m for m in config_data['commonModels'] if m['name'] == name)
        print(f"  {key}. {name} ({model_detail['params']}B 参数, {model_detail['context']} 上下文长度)")
    print("  0. 自定义模型参数")

    model_choice_key = get_user_input("请按编号选择模型 (输入 0 进行自定义): ", str, choices=list(model_options.keys()) + ['0'], default='0')

    model_size_to_use = 7.0 # 默认值
    context_length_to_use = 2048 # 默认值

    if model_choice_key != '0' and model_choice_key in model_options:
        selected_model_name = model_options[model_choice_key]
        selected_model_config = next(m for m in config_data['commonModels'] if m['name'] == selected_model_name)
        model_size_to_use = selected_model_config['params']
        default_context_length = selected_model_config['context']
        print(f"\n已选择预定义模型: {selected_model_name} ({model_size_to_use}B 参数, 默认上下文长度: {default_context_length})")
        
        use_default_context = get_bool_input(f"是否使用默认上下文长度 ({default_context_length})?", default_val_for_prompt='y', default_return_val=True)
        if use_default_context:
            context_length_to_use = default_context_length
        else:
            context_length_to_use = get_user_input(f"请输入自定义上下文长度 (默认为 {default_context_length}): ", int, default=default_context_length)
    else:
        print("\n请输入自定义模型参数:")
        model_size_to_use = get_user_input("模型参数大小 (单位 Billion, 例如 7 表示 7B): ", float, default=7.0)
        context_length_to_use = get_user_input("上下文长度 (tokens, 例如 2048): ", int, default=2048)

    data_type_choices = list(DATA_TYPE_SIZES.keys())
    print("\n可用的数据精度类型 (Data Type):")
    for i, dt in enumerate(data_type_choices):
        print(f"  {i+1}. {dt}")
    
    default_dt_idx = data_type_choices.index('float16') + 1
    data_type_idx_prompt = f"请按编号选择数据精度类型 (默认为 {default_dt_idx}. {data_type_choices[default_dt_idx-1]}): "
    data_type_idx = get_user_input(data_type_idx_prompt, int, choices=range(1, len(data_type_choices) + 1), default=default_dt_idx)
    data_type = data_type_choices[data_type_idx-1]

    batch_size = get_user_input("请输入 Batch Size (默认为 1): ", int, default=1)
    use_paged_attention = get_bool_input("是否使用 PagedAttention 优化?", default_val_for_prompt='n', default_return_val=False)
    use_tensor_parallelism = get_bool_input("是否使用 Tensor Parallelism?", default_val_for_prompt='n', default_return_val=False)
    num_gpus = 1
    if use_tensor_parallelism:
        num_gpus = get_user_input("请输入 GPU 数量 (用于 Tensor Parallelism, 默认为 1): ", int, default=1)
    is_training = get_bool_input("是否为训练模式 (计算包含 activations 的显存)?", default_val_for_prompt='n', default_return_val=False)

    return {
        "model_size_params": model_size_to_use,
        "data_type": data_type,
        "context_length": context_length_to_use,
        "batch_size": batch_size,
        "use_paged_attention": use_paged_attention,
        "use_tensor_parallelism": use_tensor_parallelism,
        "num_gpus": num_gpus,
        "is_training": is_training
    }

def display_results(vram_estimation, config_data, num_gpus_for_calc, tensor_parallelism_enabled):
    print(f"\n预估 VRAM: {vram_estimation['total_gb']:.2f} GB")
    print("显存占用分布明细:")
    # 专业词汇保持英文，但标签可以翻译
    breakdown_labels = {
        "model_weights": "模型权重 (Model Weights)",
        "kv_cache": "KV Cache",
        "activations": "激活项 (Activations)",
        "overhead": "额外开销 (Overhead)"
    }
    for component, value_gb in vram_estimation['breakdown_gb'].items():
        label = breakdown_labels.get(component, component.replace('_', ' ').capitalize())
        print(f"  - {label}: {value_gb:.2f} GB")

    if vram_estimation['total_gb'] > 0:
        print("\n推荐 GPU (基于可用 VRAM):")
        suitable_gpus = []
        for gpu in config_data['gpuModels']:
            required_vram_per_gpu = vram_estimation['total_gb'] / num_gpus_for_calc if tensor_parallelism_enabled and num_gpus_for_calc > 0 else vram_estimation['total_gb']
            if gpu['vram'] >= required_vram_per_gpu:
                suitable_gpus.append(f"{gpu['name']} ({gpu['vram']} GB)")
        
        if suitable_gpus:
            for gpu_info in suitable_gpus:
                print(f"  - {gpu_info}")
        else:
            print("  列表中没有单个 GPU 具备足够的 VRAM (或在 Tensor Parallelism 模式下均分后的 VRAM)。")
            if tensor_parallelism_enabled and num_gpus_for_calc > 1:
                 print(f"  在启用 {num_gpus_for_calc} 卡 Tensor Parallelism 的情况下，每张 GPU 大约需要 {vram_estimation['total_gb'] / num_gpus_for_calc :.2f} GB VRAM。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大型语言模型 VRAM 计算器。不带参数运行时进入交互模式。")
    config_data = load_config()
    model_choices = [model['name'] for model in config_data['commonModels']]

    parser.add_argument("--model-name", type=str, choices=model_choices, help="选择一个预定义的模型。")
    parser.add_argument("--model-size", type=float, help="模型参数大小 (单位 Billion, 例如 7B)。")
    parser.add_argument("--data-type", type=str, choices=DATA_TYPE_SIZES.keys(), help="模型权重的精度/数据类型 (例如 float16)。")
    parser.add_argument("--context-length", type=int, help="上下文长度 (tokens)。")
    parser.add_argument("--batch-size", type=int, help="Batch size。")
    parser.add_argument("--paged-attention", action="store_true", help="启用 PagedAttention 优化。")
    parser.add_argument("--tensor-parallelism", action="store_true", help="启用 Tensor Parallelism。")
    parser.add_argument("--num-gpus", type=int, help="GPU 数量 (启用 Tensor Parallelism 时使用)。")
    parser.add_argument("--training", action="store_true", help="计算训练所需的 VRAM (会包含 activations)。")

    args = parser.parse_args()
    is_cmd_line_mode = len(sys.argv) > 1 and any(arg not in [sys.argv[0], None] for arg in vars(args).values()) 
    # 更精确地判断是否有实际参数被传递，而不仅仅是脚本名
    # 检查是否有非None的args值 (action='store_true'的参数在未指定时为False, 指定时为True)
    # 或者更简单：如果除了脚本名外还有其他参数，或者某些关键参数被设置了
    # 修正判断逻辑：如果用户只输入脚本名，vars(args)中所有值都是其默认值（None或False）
    # 我们需要检查是否有任何一个参数被用户显式设置了
    # 一个简单的方法是检查 sys.argv 长度，但 argparse 会给所有参数赋默认值
    # 重新定义 is_cmd_line_mode 的判断逻辑
    # 如果除了程序名，还有其他参数，或者某些特定参数被设置了，则认为是命令行模式
    # 如下的判断方式更准确：检查是否有任何一个通过命令行设置的参数
    # (即其值不是 argparse 设置的默认值，或者对于 action='store_true' 的参数，其值为 True)
    
    # 简化判断：如果用户提供了任何一个我们关心的参数，就认为是命令行模式
    # 否则，如果只运行 `python script.py`，则进入交互模式
    provided_cmd_args = {k: v for k, v in vars(args).items() if v is not None and not (isinstance(v, bool) and v is False)} 
    # 过滤掉未设置的（None）和 action='store_true'但未指定的（False）
    is_cmd_line_mode = bool(provided_cmd_args)

    params = {}

    if is_cmd_line_mode:
        # 命令行模式
        model_size_to_use = args.model_size if args.model_size is not None else 7.0 
        context_length_to_use = args.context_length if args.context_length is not None else 2048
        
        if args.model_name:
            selected_model_config = next((m for m in config_data['commonModels'] if m['name'] == args.model_name), None)
            if selected_model_config:
                model_size_to_use = selected_model_config['params']
                # 如果命令行指定了预定义模型，也允许通过 --context-length 覆盖其默认上下文长度
                if args.context_length is None: # 仅当用户未显式指定 context-length 时，才使用模型的默认值
                    context_length_to_use = selected_model_config['context']
                print(f"使用预定义模型: {args.model_name} ({selected_model_config['params']}B 参数, 上下文长度: {context_length_to_use})")
            else:
                print(f"警告：预定义模型 '{args.model_name}' 未找到。将使用您提供或默认的参数值。")
        
        params = {
            "model_size_params": model_size_to_use,
            "data_type": args.data_type if args.data_type else "float16",
            "context_length": context_length_to_use,
            "batch_size": args.batch_size if args.batch_size is not None else 1,
            "use_paged_attention": args.paged_attention,
            "use_tensor_parallelism": args.tensor_parallelism,
            "num_gpus": args.num_gpus if args.num_gpus is not None else 1,
            "is_training": args.training
        }
    else:
        # 交互模式
        params = interactive_mode(config_data)

    vram_estimation = calculate_vram(**params)
    display_results(vram_estimation, config_data, params["num_gpus"], params["use_tensor_parallelism"])