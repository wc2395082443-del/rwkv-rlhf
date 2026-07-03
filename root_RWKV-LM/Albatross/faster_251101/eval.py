########################################################################################################
#
# RWKV GSM8K Benchmark Test
#
########################################################################################################
import os
# 禁用 CUDA 编译信息输出
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, json, re, time
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息
from tqdm import tqdm
from torch.nn import functional as F

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = "C:\\RWKV-LM\\rwkv7-g1b-1.5b-20251202-ctx8192"

print(f'\n使用 CUDA fp16 加载模型 {args.MODEL_NAME} ...\n')

from reference.rwkv7 import RWKV_x070
model = RWKV_x070(args)

from reference.utils import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("C:\\RWKV-LM\\rwkv_vocab_v20230424.txt")

########################################################################################################

# 推荐的解码参数
DECODE_PARAMS = {
    'temperature': 0.3,
    'top_k': 500,
    'top_p': 0.4,
    'alpha_presence': 0.5,
    'alpha_frequency': 0.1,
    'alpha_decay': 0.99
}

MAX_NEW_TOKENS = 512  # 最大生成token数
STOP_TOKENS = ["\n\nUser:", "\n\nQuestion:", "Q:", "<|endoftext|>"]  # 停止标记

########################################################################################################

def sample_logits(logits, temperature=1.0, top_p=0.9, top_k=0):
    """
    采样函数
    """
    probs = F.softmax(logits.float(), dim=-1)
    
    # Top-k 采样
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()
    
    # Top-p (nucleus) 采样
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()
    
    # Temperature 采样
    if temperature != 1.0:
        probs = torch.pow(probs, 1.0 / temperature)
        probs = probs / probs.sum()
    
    # 采样
    token = torch.multinomial(probs, num_samples=1)
    return token

########################################################################################################
# 答案提取和验证逻辑
########################################################################################################

def extract_answer(text):
    """
    从模型输出中提取答案 - 针对 \\boxed{} 格式优化
    
    Args:
        text: 模型输出的文本
    
    Returns:
        str: 提取的答案，如果提取失败返回 None
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # 1. 优先匹配 \boxed{数字}
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',           # 标准格式
        r'\\boxed\s*\{([^}]+)\}',        # 允许空格
        r'boxed\{([^}]+)\}',             # 缺少反斜杠
        r'\{([0-9,\.\-\s]+)\}',          # 只有花括号
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 取最后一个匹配（如果有多个）
            answer = matches[-1].strip()
            # 从答案中提取数字
            numbers = re.findall(r'-?\d+\.?\d*', answer.replace(',', ''))
            if numbers:
                return numbers[-1]  # 返回最后一个数字
    
    # 2. 尝试匹配 #### 后面的答案（GSM8K标准格式）
    if '####' in text:
        after_hash = text.split('####')[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', after_hash.replace(',', ''))
        if numbers:
            return numbers[0]
    
    # 3. 如果没有找到 boxed，尝试提取最后一行的数字
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        numbers = re.findall(r'-?\d+\.?\d*', last_line.replace(',', ''))
        if numbers:
            return numbers[-1]
    
    # 4. 兜底：提取全文最后一个数字
    all_numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if all_numbers:
        return all_numbers[-1]
    
    return None


def normalize_answer(answer):
    """
    标准化答案格式
    
    Args:
        answer: 原始答案字符串或数字
    
    Returns:
        str: 标准化后的答案字符串
    """
    if answer is None:
        return None
    
    # 清理常见格式字符
    answer_str = str(answer).strip()
    answer_str = answer_str.replace(',', '').replace('$', '').replace('%', '')
    answer_str = answer_str.replace('\\', '').replace('{', '').replace('}', '')
    
    # 移除可能的文本后缀（如 "dollars", "meters" 等）
    answer_str = re.sub(r'[a-zA-Z\s]+$', '', answer_str).strip()
    
    # 尝试转换为数字
    try:
        num = float(answer_str)
        
        # 处理整数（460.0 -> 460）
        if abs(num - round(num)) < 1e-9:
            return str(int(round(num)))
        
        # 处理小数（保留必要精度）
        formatted = f"{num:.10f}".rstrip('0').rstrip('.')
        return formatted
        
    except (ValueError, TypeError):
        # 无法转换为数字，返回清理后的字符串
        return answer_str.strip()


def compare_answers(pred, gold, tolerance=1e-6):
    """
    比较两个答案是否相等
    
    Args:
        pred: 预测答案
        gold: 正确答案
        tolerance: 数值比较的容差
    
    Returns:
        bool: 是否相等
    """
    if pred is None or gold is None:
        return False
    
    # 标准化
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm is None or gold_norm is None:
        return False
    
    # 1. 字符串完全匹配
    if pred_norm == gold_norm:
        return True
    
    # 2. 数值比较
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        
        # 绝对误差
        abs_diff = abs(pred_num - gold_num)
        if abs_diff < tolerance:
            return True
        
        # 相对误差（避免除零）
        if abs(gold_num) > tolerance:
            rel_diff = abs_diff / abs(gold_num)
            if rel_diff < tolerance:
                return True
        
        return False
        
    except (ValueError, TypeError):
        pass
    
    # 3. 大小写不敏感的字符串比较
    return pred_norm.lower() == gold_norm.lower()


def verify_gsm8k_answer(model_response, correct_answer, verbose=False):
    """
    验证 GSM8K 答案
    
    Args:
        model_response: 模型的完整回复
        correct_answer: 正确答案
        verbose: 是否输出详细信息
    
    Returns:
        dict: 验证结果
    """
    # 提取答案
    extracted = extract_answer(model_response)
    
    # 标准化
    pred_norm = normalize_answer(extracted)
    gold_norm = normalize_answer(correct_answer)
    
    # 比较
    is_correct = compare_answers(pred_norm, gold_norm)
    
    result = {
        'is_correct': is_correct,
        'extracted_answer': extracted,
        'normalized_pred': pred_norm,
        'normalized_gold': gold_norm,
        'raw_response': model_response[:200] if model_response else None
    }
    
    if verbose:
        print(f"Raw response: {model_response[:100]}...")
        print(f"Extracted: {extracted}")
        print(f"Pred (normalized): {pred_norm}")
        print(f"Gold (normalized): {gold_norm}")
        print(f"Match: {is_correct}")
    
    return result

########################################################################################################

def generate_response(prompt, state, max_tokens=MAX_NEW_TOKENS):
    """
    使用推荐参数生成回复
    """
    all_tokens = []
    out = model.forward(tokenizer.encode(prompt), state)
    
    # 用于频率惩罚的计数器
    occurrence = {}
    
    for i in range(max_tokens):
        # 应用频率和存在惩罚
        for token_id, count in occurrence.items():
            out[token_id] -= (
                DECODE_PARAMS['alpha_presence'] + 
                count * DECODE_PARAMS['alpha_frequency']
            )
        
        # 使用采样函数
        token = sample_logits(
            out, 
            temperature=DECODE_PARAMS['temperature'],
            top_p=DECODE_PARAMS['top_p'],
            top_k=DECODE_PARAMS['top_k']
        ).item()
        
        all_tokens.append(token)
        
        # 更新token出现次数
        occurrence[token] = occurrence.get(token, 0) + 1
        # 应用衰减
        for k in occurrence:
            occurrence[k] *= DECODE_PARAMS['alpha_decay']
        
        # 检查是否遇到停止标记
        try:
            current_text = tokenizer.decode(all_tokens, utf8_errors="strict")
            if any(stop in current_text for stop in STOP_TOKENS):
                break
        except:
            pass
        
        out = model.forward([token], state)
    
    # 解码最终文本
    try:
        response = tokenizer.decode(all_tokens, utf8_errors="replace")
    except:
        response = ""
    
    return response

########################################################################################################

def build_prompt(problem: str) -> str:
    """
    构建GSM8K问题的prompt
    要求模型将答案放在 \\boxed{} 中
    """
    p = (problem or "").strip()
    return (
        f"User: {p}\n"
        f"请将最终答案放在\\boxed{{...}}里，并且最终只给出\\boxed{{...}}这一行，不要输出多余内容。\n"
        f"Assistant: "
    )

########################################################################################################

def test_gsm8k(data_file="gsm8k_test_formatted.jsonl", num_samples=None):
    """
    测试RWKV在GSM8K上的表现
    
    Args:
        data_file: GSM8K测试数据文件路径
        num_samples: 测试样本数量，None表示测试全部
    """
    print(f"\n{'#'*80}")
    print(f"# 开始GSM8K Benchmark测试")
    print(f"# 模型: {args.MODEL_NAME}")
    print(f"# 解码参数: {DECODE_PARAMS}")
    print(f"# 最大生成长度: {MAX_NEW_TOKENS}")
    print(f"{'#'*80}\n")
    
    # 加载测试数据
    problems = []
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_file}")
        return None, []
    
    if num_samples:
        problems = problems[:num_samples]
    
    print(f"加载了 {len(problems)} 个测试问题\n")
    
    # 测试结果
    correct = 0
    total = 0
    results = []
    
    # 测试每个问题
    for idx, problem in enumerate(tqdm(problems, desc="测试进度")):
        question = problem.get('problem', '') or problem.get('question', '')
        solution = problem.get('solution', '') or problem.get('answer', '')
        
        # 从solution中提取正确答案
        correct_answer = extract_answer(solution)
        
        # 构建prompt
        prompt = build_prompt(question)
        
        # 生成回答
        try:
            state = model.generate_zero_state(0)
            response = generate_response(prompt, state)
        except Exception as e:
            print(f"\n生成回答时出错 (问题 {idx}): {e}")
            response = ""
        
        # 验证答案
        verify_result = verify_gsm8k_answer(response, correct_answer)
        
        is_correct = verify_result['is_correct']
        if is_correct:
            correct += 1
        total += 1
        
        # 保存结果
        result = {
            'index': idx,
            'question': question,
            'correct_answer': verify_result['normalized_gold'],
            'model_answer': verify_result['normalized_pred'],
            'extracted_raw': verify_result['extracted_answer'],
            'is_correct': is_correct,
            'full_response': response
        }
        results.append(result)
        
        # 每10个问题打印一次中间结果
        if (idx + 1) % 10 == 0:
            current_acc = correct / total * 100
            print(f"\n当前准确率 ({total} 题): {current_acc:.2f}% ({correct}/{total})")
    
    # 计算最终准确率
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n{'#'*80}")
    print(f"# 测试完成!")
    print(f"# 总题数: {total}")
    print(f"# 正确数: {correct}")
    print(f"# 准确率: {accuracy:.2f}%")
    print(f"{'#'*80}\n")
    
    # 保存详细结果
    output_file = f"gsm8k_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': args.MODEL_NAME,
            'decode_params': DECODE_PARAMS,
            'max_new_tokens': MAX_NEW_TOKENS,
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"详细结果已保存到: {output_file}\n")
    
    # 显示一些错误案例
    print("=" * 80)
    print("错误案例示例 (前5个):")
    print("=" * 80)
    error_count = 0
    for result in results:
        if not result['is_correct'] and error_count < 5:
            print(f"\n问题 {result['index']}:")
            print(f"题目: {result['question'][:100]}...")
            print(f"正确答案: {result['correct_answer']}")
            print(f"模型答案: {result['model_answer']}")
            print(f"提取原始: {result['extracted_raw']}")
            print(f"模型回复: {result['full_response'][:200]}...")
            print("-" * 80)
            error_count += 1
    
    # 显示一些正确案例
    print("\n" + "=" * 80)
    print("正确案例示例 (前3个):")
    print("=" * 80)
    correct_count = 0
    for result in results:
        if result['is_correct'] and correct_count < 3:
            print(f"\n问题 {result['index']}:")
            print(f"题目: {result['question'][:100]}...")
            print(f"正确答案: {result['correct_answer']}")
            print(f"模型答案: {result['model_answer']}")
            print(f"模型回复: {result['full_response'][:200]}...")
            print("-" * 80)
            correct_count += 1
    
    return accuracy, results

########################################################################################################

if __name__ == "__main__":
    # 运行测试
    #print("开始快速测试 (100个样本)...")
    #accuracy, results = test_gsm8k(num_samples=100)
    
    # 如果需要完整测试，取消下面的注释
    print("\n开始完整测试...")
    accuracy, results = test_gsm8k()


