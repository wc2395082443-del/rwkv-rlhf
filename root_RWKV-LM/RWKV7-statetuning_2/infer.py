#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional
import re
import torch
import torch.nn.functional as F


# 停止标记
STOP_TOKENS = ["User:", "Assistant:", "\n\nUser", "\n\nAssistant", "<|endoftext|>"]


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    presence_penalty: float = 0.5,
    frequency_penalty: float = 0.1
) -> torch.Tensor:
    """
    应用重复惩罚 (Repetition Penalty)
    
    Args:
        logits: [B, vocab_size] 原始logits
        token_counts: [B, vocab_size] 每个token出现的次数
        presence_penalty: 存在惩罚 (只要出现过就惩罚)
        frequency_penalty: 频率惩罚 (根据出现次数惩罚)
    
    Returns:
        惩罚后的logits
    """
    if presence_penalty <= 0 and frequency_penalty <= 0:
        return logits
    
    # 计算惩罚
    mask = (token_counts > 0).float()  # 是否出现过
    penalty = (mask * presence_penalty) + (token_counts * frequency_penalty)
    
    # 应用惩罚
    return logits - penalty


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用温度缩放
    
    Args:
        logits: [B, vocab_size]
        temperature: 温度参数
    
    Returns:
        缩放后的logits
    """
    if temperature <= 0 or temperature == 1.0:
        return logits
    
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    应用Top-K过滤
    
    Args:
        logits: [B, vocab_size]
        top_k: 保留的top k个候选
    
    Returns:
        过滤后的logits
    """
    if top_k <= 0:
        return logits
    
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    # 将不在top-k中的logits设为-inf
    logits[logits < v[:, [-1]]] = float('-inf')
    
    return logits


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    应用Top-P (Nucleus) 过滤
    
    Args:
        probs: [B, vocab_size] 概率分布
        top_p: nucleus参数
    
    Returns:
        过滤后的概率分布
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return probs
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过top_p的位置
    sorted_indices_to_remove = cumulative_probs > top_p
    # 至少保留一个候选
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    
    # 映射回原始索引
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    
    # 过滤并重新归一化
    probs = probs.clone()
    probs[indices_to_remove] = 0.0
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
    
    return probs


def sample_next_token(
    logits: torch.Tensor,
    token_counts: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float = 0.5,
    frequency_penalty: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    采样下一个token
    
    正确的顺序: 温度 -> 计算概率 -> Top-P -> Top-K -> 采样
    
    Args:
        logits: [B, vocab_size]
        token_counts: [B, vocab_size] token计数
        temperature: 温度
        top_p: nucleus参数
        top_k: top-k参数
        presence_penalty: 存在惩罚
        frequency_penalty: 频率惩罚
    
    Returns:
        (token_ids, log_probs): 采样的token和其对应的log概率
    """
    # 保存原始logits用于计算log_prob
    original_logp = F.log_softmax(logits.float(), dim=-1)
    
    # 步骤1: 应用温度
    logits = apply_temperature(logits, temperature)
    
    # 步骤2: 计算概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 步骤3: 应用Top-P
    probs = apply_top_p(probs, top_p)
    
    # 步骤4: 应用Top-K (在概率上)
    if top_k > 0:
        k = min(top_k, probs.size(-1))
        v, idx = torch.topk(probs, k)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, idx, 1.0)
        probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
    
    # 步骤5: 采样
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # 获取原始log概率 (用于RL训练)
    log_probs = original_logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_ids, log_probs


class AlbatrossBatchInference:
    """批量推理引擎"""
    
    def __init__(self, infer_model, train_model, encode_fn, decode_fn, device: str, cfg):
        self.infer_model = infer_model
        self.train_model = train_model
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        self.cfg = cfg
    
    def init_state_with_time_state(self, B: int):
        """初始化状态，使用训练模型的time_state"""
        state = self.infer_model.generate_zero_state(B)
        for i, block in enumerate(self.train_model.blocks):
            ts = block.att.time_state
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone()
        return state
    
    @torch.no_grad()
    def prime_prompts(self, prompt_tokens_list: List[List[int]]):
        """处理初始提示"""
        B = len(prompt_tokens_list)
        state = self.init_state_with_time_state(B)
        out = self.infer_model.forward_batch(prompt_tokens_list, state)
        if torch.is_tensor(out) and out.dim() == 3:
            out = out[:, -1, :]
        return out, state
    
    @torch.no_grad()
    def generate_group_parallel(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_on_think_close: bool = False,
        stop_on_user: bool = True,
        stop_on_boxed: bool = False,
        stop_on_repeat_ngram: bool = True,
        repeat_ngram_n: int = 16,
        repeat_ngram_repeat: int = 5,
        presence_penalty: float = 0.5,
        frequency_penalty: float = 0.1,
        alpha_decay: float = 0.99,
        post_trunc_append: str = " so the answer is : ",
        post_trunc_max_tokens: int = 64,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:
        """
        并行生成多组响应
        
        Args:
            prompt_tokens_list: 提示token列表
            group_size: 每个提示生成的响应数
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: nucleus参数
            top_k: top-k参数
            stop_on_think_close: 遇到</think>时停止
            stop_on_user: 遇到User:时停止
            stop_on_boxed: 遇到完整的\\boxed{}时停止
            presence_penalty: 存在惩罚
            frequency_penalty: 频率惩罚
            alpha_decay: 惩罚衰减系数
            post_trunc_append: 截断后强制追加的文本（计入训练）
            post_trunc_max_tokens: 截断后继续生成的最大token数（计入训练）
        
        Returns:
            (comp_tokens, log_probs, comp_texts, truncated)
        """
        Bp = len(prompt_tokens_list)
        if Bp == 0:
            return [], [], [], []
        
        # 1. 处理提示
        last_logits, state = self.prime_prompts(prompt_tokens_list)
        
        # 2. 复制状态以支持group_size
        B = Bp * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()
        
        # 复制RWKV7状态
        state0 = state[0].repeat_interleave(group_size, dim=2).contiguous()
        state1 = state[1].repeat_interleave(group_size, dim=1).contiguous()
        state = [state0, state1]
        
        # 3. 初始化生成状态
        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        log_probs: List[List[float]] = [[] for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]
        
        # token计数器 (用于重复惩罚)
        token_counts = torch.zeros((B, last_logits.size(-1)), device=last_logits.device)
        
        # 4. 生成循环
        for t in range(max_new_tokens):
            if not active.any():
                break
            
            # 采样下一个token
            token_ids, picked_logp = sample_next_token(
                logits=last_logits,
                token_counts=token_counts,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            
            # 更新token计数
            token_counts.scatter_add_(
                1, 
                token_ids.view(-1, 1), 
                torch.ones_like(token_ids.view(-1, 1), dtype=torch.float32)
            )
            # 衰减历史计数
            token_counts *= alpha_decay
            
            # 掩码处理
            token_ids = torch.where(active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(active, picked_logp, torch.zeros_like(picked_logp))
            
            # 保存结果
            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()
            
            for i in range(B):
                if active[i]:
                    comp_tokens[i].append(int(tok_cpu[i]))
                    log_probs[i].append(float(lp_cpu[i]))
            
            # 停止条件检查
            if t % 10 == 0:  # 每10步检查一次
                for i in range(B):
                    if not active[i]:
                        continue
                    
                    # 解码当前文本
                    text = self.decode(comp_tokens[i])
                    
                    # 检查停止条件
                    if stop_on_think_close and "</think>" in text:
                        active[i] = False
                    elif stop_on_user and any(tok in text for tok in STOP_TOKENS):
                        active[i] = False
                    elif stop_on_boxed and self._check_boxed_complete(text):
                        active[i] = False
                    elif stop_on_repeat_ngram and self._has_repeated_ngrams(text, n=repeat_ngram_n, repeat=repeat_ngram_repeat):
                        active[i] = False
            
            # 下一步推理
            step_tokens_batch = [[int(x)] for x in tok_cpu]
            last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]
        
        # 5. 标记截断的序列
        for i in range(B):
            if active[i]:
                truncated[i] = True

        # 6. 截断后强制追加尾巴并继续生成（计入训练）
        if any(truncated) and post_trunc_append and post_trunc_max_tokens > 0:
            tail_prefix = self.encode(post_trunc_append)
            if tail_prefix:
                for tok in tail_prefix:
                    step_tokens = [int(tok) if truncated[i] else 0 for i in range(B)]
                    token_ids = torch.tensor(step_tokens, device=last_logits.device, dtype=torch.long)

                    # 计算强制token的logp
                    logp_all = F.log_softmax(last_logits, dim=-1)
                    picked_logp = logp_all.gather(1, token_ids.view(-1, 1)).squeeze(1)

                    token_counts.scatter_add_(
                        1,
                        token_ids.view(-1, 1),
                        torch.ones_like(token_ids.view(-1, 1), dtype=torch.float32)
                    )
                    token_counts *= alpha_decay

                    tok_cpu = token_ids.detach().cpu().tolist()
                    lp_cpu = picked_logp.detach().cpu().tolist()
                    for i in range(B):
                        if truncated[i]:
                            comp_tokens[i].append(int(tok_cpu[i]))
                            log_probs[i].append(float(lp_cpu[i]))

                    step_tokens_batch = [[int(x)] for x in step_tokens]
                    last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
                    if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                        last_logits = last_logits[:, -1, :]

            active_tail = torch.tensor(truncated, device=last_logits.device, dtype=torch.bool)
            for t in range(post_trunc_max_tokens):
                if not active_tail.any():
                    break

                token_ids, picked_logp = sample_next_token(
                    logits=last_logits,
                    token_counts=token_counts,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )

                token_ids = torch.where(active_tail, token_ids, torch.zeros_like(token_ids))
                picked_logp = torch.where(active_tail, picked_logp, torch.zeros_like(picked_logp))

                token_counts.scatter_add_(
                    1,
                    token_ids.view(-1, 1),
                    torch.ones_like(token_ids.view(-1, 1), dtype=torch.float32)
                )
                token_counts *= alpha_decay

                tok_cpu = token_ids.detach().cpu().tolist()
                lp_cpu = picked_logp.detach().cpu().tolist()
                for i in range(B):
                    if active_tail[i]:
                        comp_tokens[i].append(int(tok_cpu[i]))
                        log_probs[i].append(float(lp_cpu[i]))

                if t % 10 == 0:
                    for i in range(B):
                        if not active_tail[i]:
                            continue
                        text = self.decode(comp_tokens[i])
                        if stop_on_think_close and '</think>' in text:
                            active_tail[i] = False
                        elif stop_on_user and any(tok in text for tok in STOP_TOKENS):
                            active_tail[i] = False
                        elif stop_on_boxed and self._check_boxed_complete(text):
                            active_tail[i] = False
                        elif stop_on_repeat_ngram and self._has_repeated_ngrams(text, n=repeat_ngram_n, repeat=repeat_ngram_repeat):
                            active_tail[i] = False

                step_tokens_batch = [[int(x)] for x in tok_cpu]
                last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
                if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                    last_logits = last_logits[:, -1, :]

        # 7. 解码文本
        comp_texts = []
        for i in range(B):
            text = self.decode(comp_tokens[i])
            comp_texts.append(text)

        return comp_tokens, log_probs, comp_texts, truncated
    
    def _has_repeated_ngrams(self, text: str, n: int = 16, repeat: int = 5) -> bool:
        if not text or n <= 0 or repeat <= 1:
            return False
        tokens = re.findall(r'\w+|[^\w\s]', text)
        total = n * repeat
        if len(tokens) < total:
            return False
        counts = {}
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i + n])
            cnt = counts.get(ng, 0) + 1
            counts[ng] = cnt
            if cnt >= repeat:
                return True
        return False

    def _check_boxed_complete(self, text: str) -> bool:
        """检查是否有完整的\\boxed{}"""
        k = text.find(r"\boxed{")
        if k < 0:
            return False
        i = k + len(r"\boxed{")
        depth = 1
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return True
            i += 1
        return False