########################################################################################################
#
# RWKV-7 GSM8K Mathematical Reasoning Evaluation - FIXED
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, json, re, os, sys, time
from tqdm import tqdm
from torch.nn import functional as F
import random

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################
# 配置参数
########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = "C:\\RWKV-LM\\rwkv7-g1b-1.5b-20251202-ctx8192"

# GSM8K 数据集路径
GSM8K_PATH = "C:\\RWKV-LM\\RWKV7-statetuning\\gsm8k_test_formatted.jsonl"

# Tokenizer 路径
TOKENIZER_PATH = "C:\\RWKV-LM\\rwkv_vocab_v20230424.txt"

# 推荐的解码参数
DECODE_PARAMS = {
    'temperature': 1.0,      # 先用默认值测试
    'top_k': 0,
    'top_p': 0.85,
    'alpha_presence': 0.3,
    'alpha_frequency': 0.3,
    'alpha_decay': 0.996
}

# 评估参数
MAX_SAMPLES = 10            # 先测试 10 个样本
MAX_NEW_TOKENS = 512
USE_EARLY_STOPPING = False  # 先关闭早停，看看生成效果

# 早停触发词
STOP_TOKENS = ['####', '\\boxed{', '\n\n\n', 'Question:']

print(f'\n{"="*80}')
print(f'RWKV-7 GSM8K Mathematical Reasoning Evaluation')
print(f'{"="*80}')
print(f'Model: {args.MODEL_NAME}')
print(f'Dataset: {GSM8K_PATH}')
print(f'Tokenizer: {TOKENIZER_PATH}')
print(f'Decode Params: {DECODE_PARAMS}')
print(f'Max Samples: {MAX_SAMPLES if MAX_SAMPLES else "All"}')
print(f'{"="*80}\n')

########################################################################################################
# 加载 Tokenizer
########################################################################################################

print(f'Loading tokenizer from: {TOKENIZER_PATH}')

tokenizer = None
current_dir = os.path.dirname(os.path.abspath(__file__))
reference_dir = os.path.join(current_dir, 'reference')

# 方法 1: 从 reference 目录加载 TRIE_TOKENIZER
if os.path.exists(reference_dir):
    sys.path.insert(0, reference_dir)
    try:
        from utils import TRIE_TOKENIZER
        if os.path.exists(TOKENIZER_PATH):
            tokenizer = TRIE_TOKENIZER(TOKENIZER_PATH)
            print("✓ Tokenizer loaded successfully (TRIE_TOKENIZER)")
        else:
            ref_tokenizer_path = os.path.join(reference_dir, "rwkv_vocab_v20230424.txt")
            if os.path.exists(ref_tokenizer_path):
                tokenizer = TRIE_TOKENIZER(ref_tokenizer_path)
                TOKENIZER_PATH = ref_tokenizer_path
                print(f"✓ Tokenizer loaded from: {ref_tokenizer_path}")
    except ImportError as e:
        print(f"⚠ Could not load TRIE_TOKENIZER: {e}")

# 方法 2: 使用标准 RWKV tokenizer
if tokenizer is None:
    try:
        from rwkv.utils import TOKENIZER
        tokenizer = TOKENIZER(TOKENIZER_PATH)
        print("✓ Tokenizer loaded successfully (RWKV TOKENIZER)")
    except ImportError as e:
        print(f"⚠ Could not load RWKV TOKENIZER: {e}")

if tokenizer is None:
    raise RuntimeError("Failed to load tokenizer. Please check the tokenizer path and installation.")

# 测试 tokenizer
test_text = "Hello, world!"
test_tokens = tokenizer.encode(test_text)
test_decoded = tokenizer.decode(test_tokens)
print(f"Tokenizer test: '{test_text}' -> {test_tokens} -> '{test_decoded}'")
print()

########################################################################################################
# 加载模型
########################################################################################################

if os.path.exists(reference_dir):
    sys.path.insert(0, reference_dir)

try:
    from rwkv7 import RWKV_x070
    print("✓ Using RWKV7 model")
except ImportError:
    print("✗ Failed to import RWKV7 model")
    sys.exit(1)

print(f'Loading model...')
model = RWKV_x070(args)
print(f'✓ Model loaded successfully!\n')

########################################################################################################
# 采样函数
########################################################################################################

def sample_logits(logits, temperature=1.0, top_k=0, top_p=0.0, 
                  alpha_presence=0.0, alpha_frequency=0.0, token_counts=None):
    """高级采样函数"""
    logits = logits.float()
    
    # 应用 presence 和 frequency penalty
    if token_counts is not None:
        for token_id, count in token_counts.items():
            if alpha_presence > 0:
                logits[token_id] -= alpha_presence
            if alpha_frequency > 0:
                logits[token_id] -= alpha_frequency * count
    
    # 应用温度
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    
    # Top-K filtering
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float('-inf')
    
    # Top-P (nucleus) filtering
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    
    # 确保概率和为1
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        print("⚠ Warning: Invalid probabilities detected, using greedy decoding")
        token = torch.argmax(logits).item()
    else:
        token = torch.multinomial(probs, num_samples=1).item()
    
    return token

########################################################################################################
# 答案提取函数
########################################################################################################

def extract_answer_from_text(text):
    """从生成的文本中提取数字答案"""
    if not text:
        return None
    
    # 方法 1: 提取 \boxed{} 中的内容
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.findall(boxed_pattern, text)
    if boxed_match:
        try:
            answer_str = boxed_match[-1].replace(',', '').strip()
            numbers = re.findall(r'-?\d+\.?\d*', answer_str)
            if numbers:
                num_str = numbers[0]
                return float(num_str) if '.' in num_str else int(num_str)
        except:
            pass
    
    # 方法 2: 提取 #### 后的数字
    hash_pattern = r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    hash_match = re.findall(hash_pattern, text)
    if hash_match:
        try:
            answer_str = hash_match[-1].replace(',', '')
            return float(answer_str) if '.' in answer_str else int(answer_str)
        except:
            pass
    
    # 方法 3: 提取 "answer is" 后的数字
    answer_patterns = [
        r'(?:the answer is|answer is|answer:|答案是|答案：)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(?:total|sum|result)(?:\s+is)?\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                answer_str = matches[-1].replace(',', '')
                return float(answer_str) if '.' in answer_str else int(answer_str)
            except:
                pass
    
    # 方法 4: 提取最后一个数字
    all_numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if all_numbers:
        try:
            for num_str in reversed(all_numbers[-5:]):
                answer_str = num_str.replace(',', '')
                if '.' in answer_str:
                    decimal_places = len(answer_str.split('.')[1])
                    if decimal_places <= 2:
                        return float(answer_str)
                else:
                    return int(answer_str)
        except:
            pass
    
    return None

def extract_ground_truth(solution_text):
    """从 solution 字段提取标准答案"""
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, solution_text)
    if match:
        try:
            answer_str = match.group(1).replace(',', '').strip()
            numbers = re.findall(r'-?\d+\.?\d*', answer_str)
            if numbers:
                num_str = numbers[0]
                return float(num_str) if '.' in num_str else int(num_str)
        except:
            pass
    return None

def normalize_answer(answer):
    """标准化答案格式"""
    if answer is None:
        return None
    
    try:
        if isinstance(answer, str):
            answer = answer.replace(',', '').replace('$', '').strip()
            answer = float(answer)
        
        if isinstance(answer, float) and answer.is_integer():
            return int(answer)
        
        if isinstance(answer, float):
            return round(answer, 2)
        
        return answer
    except:
        return None

########################################################################################################
# 生成函数 - 修复版
########################################################################################################

def generate_response(prompt, model, tokenizer, max_new_tokens=512, 
                     temperature=1.0, top_k=0, top_p=0.0,
                     alpha_presence=0.0, alpha_frequency=0.0, alpha_decay=0.99,
                     use_early_stopping=True, stop_tokens=None, verbose=False):
    """生成模型响应 - 修复版"""
    
    # 编码输入
    input_ids = tokenizer.encode(prompt)
    
    if verbose:
        print(f"Input tokens: {len(input_ids)}")
        print(f"Input text: {prompt[:100]}...")
    
    # ✅ 关键修复：每次都创建新的 state
    state = model.generate_zero_state(0)
    
    # ✅ 关键修复：直接传入整个 token 列表
    out = model.forward(input_ids, state)
    
    # 生成新 tokens
    generated_tokens = []
    token_counts = {}
    
    current_alpha_presence = alpha_presence
    current_alpha_frequency = alpha_frequency
    
    for step in range(max_new_tokens):
        # 采样下一个 token
        next_token = sample_logits(
            out, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            alpha_presence=current_alpha_presence,
            alpha_frequency=current_alpha_frequency,
            token_counts=token_counts
        )
        
        # 更新 token 计数
        token_counts[next_token] = token_counts.get(next_token, 0) + 1
        
        # 衰减 penalty
        current_alpha_presence *= alpha_decay
        current_alpha_frequency *= alpha_decay
        
        # 添加到生成序列
        generated_tokens.append(next_token)
        
        # 解码当前生成的文本（用于早停检查）
        if use_early_stopping and stop_tokens:
            try:
                generated_text = tokenizer.decode(generated_tokens)
                for stop_token in stop_tokens:
                    if stop_token in generated_text:
                        if verbose:
                            print(f"Early stopping triggered by: {stop_token}")
                        return generated_text, len(generated_tokens)
            except:
                pass
        
        # ✅ 关键修复：单个 token 作为整数传入
        try:
            out = model.forward(next_token, state)
        except Exception as e:
            if verbose:
                print(f"⚠ Forward error at step {step}: {e}")
            break
    
    # 最终解码
    try:
        generated_text = tokenizer.decode(generated_tokens)
    except Exception as e:
        print(f"⚠ Final decode error: {e}")
        generated_text = ""
    
    if verbose:
        print(f"Generated {len(generated_tokens)} tokens")
        print(f"Generated text: {generated_text[:200]}...")
    
    return generated_text, len(generated_tokens)


########################################################################################################
# 加载 GSM8K 数据集
########################################################################################################

def load_gsm8k(file_path, max_samples=None):
    """加载 GSM8K 数据集"""
    print(f'Loading GSM8K dataset from: {file_path}')
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                    
                    if max_samples and len(data) >= max_samples:
                        break
                except json.JSONDecodeError as e:
                    print(f"⚠ Warning: Failed to parse line: {e}")
                    continue
    
    print(f'✓ Loaded {len(data)} samples\n')
    return data

########################################################################################################
# 评估函数
########################################################################################################

def evaluate_gsm8k(data, model, tokenizer, decode_params, 
                   max_new_tokens=512, use_early_stopping=True, 
                   stop_tokens=None, verbose=False):
    """评估 GSM8K 数据集"""
    
    correct = 0
    total = len(data)
    results = []
    
    total_tokens = 0
    total_time = 0
    
    print(f'Starting evaluation on {total} samples...\n')
    
    # 先测试第一个样本（详细输出）
    if total > 0 and verbose:
        print(f'\n{"="*80}')
        print(f'Testing first sample with verbose output:')
        print(f'{"="*80}\n')
        
        item = data[0]
        question = item['problem']
        prompt = f"Question: {question}\n\nAnswer: Let's solve this step by step.\n"
        
        print(f"Prompt:\n{prompt}\n")
        
        generated_text, num_tokens = generate_response(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_early_stopping=use_early_stopping,
            stop_tokens=stop_tokens,
            verbose=True,
            **decode_params
        )
        
        print(f"\nGenerated text:\n{generated_text}\n")
        print(f"Number of tokens: {num_tokens}\n")
        print(f'{"="*80}\n')
    
    for idx, item in enumerate(tqdm(data, desc="Evaluating")):
        question = item['problem']
        solution = item['solution']
        
        # 提取标准答案
        ground_truth_answer = normalize_answer(extract_ground_truth(solution))
        
        # 构建提示
        prompt = f"Question: {question}\n\nAnswer: Let's solve this step by step.\n"
        
        # 生成响应
        start_time = time.time()
        try:
            generated_text, num_tokens = generate_response(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                use_early_stopping=use_early_stopping,
                stop_tokens=stop_tokens,
                verbose=False,
                **decode_params
            )
        except Exception as e:
            print(f"\n⚠ Error generating response for sample {idx}: {e}")
            generated_text = ""
            num_tokens = 0
        
        elapsed_time = time.time() - start_time
        
        # 提取预测答案
        predicted_answer = normalize_answer(extract_answer_from_text(generated_text))
        
        # 判断是否正确
        is_correct = False
        if predicted_answer is not None and ground_truth_answer is not None:
            if isinstance(predicted_answer, float) or isinstance(ground_truth_answer, float):
                is_correct = abs(float(predicted_answer) - float(ground_truth_answer)) < 0.01
            else:
                is_correct = (predicted_answer == ground_truth_answer)
        
        if is_correct:
            correct += 1
        
        # 记录结果
        result = {
            'index': idx,
            'question': question,
            'ground_truth': ground_truth_answer,
            'predicted': predicted_answer,
            'correct': is_correct,
            'generated_text': generated_text,
            'num_tokens': num_tokens,
            'time': elapsed_time
        }
        results.append(result)
        
        total_tokens += num_tokens
        total_time += elapsed_time
        
        # 每10个样本打印一次进度
        if (idx + 1) % 10 == 0:
            current_acc = correct / (idx + 1)
            avg_tokens = total_tokens / (idx + 1)
            print(f'\nProgress: {idx + 1}/{total} | Accuracy: {current_acc:.2%} ({correct}/{idx + 1}) | Avg tokens: {avg_tokens:.1f}')
    
    # 计算统计信息
    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # 打印结果
    print(f'\n{"="*80}')
    print(f'EVALUATION RESULTS')
    print(f'{"="*80}')
    print(f'Total Samples: {total}')
    print(f'Correct: {correct}')
    print(f'Accuracy: {accuracy:.2%}')
    print(f'Average Tokens per Sample: {avg_tokens:.1f}')
    print(f'Average Time per Sample: {avg_time:.2f}s')
    print(f'Generation Speed: {tokens_per_sec:.1f} tokens/s')
    print(f'{"="*80}\n')
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_tokens': avg_tokens,
        'avg_time': avg_time,
        'tokens_per_sec': tokens_per_sec,
        'results': results
    }

########################################################################################################
# 主函数
########################################################################################################

def main():
    # 加载数据集
    data = load_gsm8k(GSM8K_PATH, max_samples=MAX_SAMPLES)
    
    if len(data) == 0:
        print("✗ No data loaded. Please check the dataset path.")
        return
    
    # 运行评估
    eval_results = evaluate_gsm8k(
        data=data,
        model=model,
        tokenizer=tokenizer,
        decode_params=DECODE_PARAMS,
        max_new_tokens=MAX_NEW_TOKENS,
        use_early_stopping=USE_EARLY_STOPPING,
        stop_tokens=STOP_TOKENS,
        verbose=True  # 启用详细输出
    )
    
    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'gsm8k_evaluation_results.json')
    
    summary = {
        'accuracy': eval_results['accuracy'],
        'correct': eval_results['correct'],
        'total': eval_results['total'],
        'avg_tokens': eval_results['avg_tokens'],
        'avg_time': eval_results['avg_time'],
        'tokens_per_sec': eval_results['tokens_per_sec'],
        'config': {
            'model': args.MODEL_NAME,
            'decode_params': DECODE_PARAMS,
            'max_new_tokens': MAX_NEW_TOKENS,
            'max_samples': MAX_SAMPLES
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f'✓ Summary saved to: {output_file}')
    
    # 保存详细结果
    detailed_output_file = os.path.join(output_dir, 'gsm8k_detailed_results.jsonl')
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        for result in eval_results['results']:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f'✓ Detailed results saved to: {detailed_output_file}')
    
    # 打印错误案例和正确案例
    print(f'\n{"="*80}')
    print(f'Sample Cases:')
    print(f'{"="*80}')
    
    correct_cases = [r for r in eval_results['results'] if r['correct']]
    error_cases = [r for r in eval_results['results'] if not r['correct']]
    
    if correct_cases:
        print(f"\n✓ Correct Case Example:")
        result = correct_cases[0]
        print(f"Question: {result['question'][:100]}...")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Generated: {result['generated_text'][:300]}...")
    
    if error_cases:
        print(f"\n✗ Error Case Example:")
        result = error_cases[0]
        print(f"Question: {result['question'][:100]}...")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Generated: {result['generated_text'][:300]}...")
    
    # 打印 kernel 统计
    if hasattr(model, 'print_kernel_stats'):
        model.print_kernel_stats()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
