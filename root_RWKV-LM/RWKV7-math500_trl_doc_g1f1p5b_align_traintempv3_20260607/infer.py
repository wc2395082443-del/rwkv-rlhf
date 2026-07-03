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
    
    正确的顺序: 重复惩罚 -> 温度 -> Top-K -> 计算概率 -> Top-P -> 采样
    
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
    # original logits log_prob
    original_logp = F.log_softmax(logits.float(), dim=-1)
    
    # use float32 for sampling path
    logits = logits.float()
    
    # 步骤1: 应用重复惩罚
    #logits = apply_repetition_penalty(
    #    logits.float(), 
    #    token_counts, 
    #    presence_penalty, 
    #    frequency_penalty
    #)
    
    # 步骤2: 应用温度
    logits = apply_temperature(logits, temperature)
    
    # 步骤3: 应用Top-K
    logits = apply_top_k(logits, top_k)
    
    # 步骤4: 计算概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 步骤5: 应用Top-P
    probs = apply_top_p(probs, top_p)
    
    # 步骤6: 采样
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # 获取采样分布log概率 (用于RL训练)
    # original log_prob (for RL)
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
        self._stop_token_id_seqs: List[List[int]] = []
        for s in STOP_TOKENS:
            try:
                ids = self.encode(s)
            except Exception:
                ids = None
            if ids:
                self._stop_token_id_seqs.append([int(x) for x in ids])
        # Match longer sequences first, e.g.  \n\nUser before User:
        self._stop_token_id_seqs.sort(key=len, reverse=True)

    def _use_full_rollout(self) -> bool:
        return getattr(self.cfg, "tune_mode", "state") == "full" and self.infer_model is None
    
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
        if self._use_full_rollout():
            return self._prime_prompts_full(prompt_tokens_list), None
        state = self.init_state_with_time_state(B)
        out = self.infer_model.forward_batch(prompt_tokens_list, state)
        if torch.is_tensor(out) and out.dim() == 3:
            out = out[:, -1, :]
        return out, state

    @torch.no_grad()
    def _forward_last_logits_full(self, seqs: List[List[int]]) -> torch.Tensor:
        batch_size = max(1, int(getattr(self.cfg, "rollout_forward_batch", 8)))
        outputs = []
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start:start + batch_size]
            max_len = max(len(x) for x in chunk)
            x = torch.zeros((len(chunk), max_len), dtype=torch.long, device=self.device)
            for i, ids in enumerate(chunk):
                x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            logits = self.train_model(x)
            last_idx = torch.tensor([len(ids) - 1 for ids in chunk], dtype=torch.long, device=self.device)
            batch_last = logits[torch.arange(len(chunk), device=self.device), last_idx, :]
            outputs.append(batch_last)
            del x, logits, last_idx, batch_last
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def _prime_prompts_full(self, prompt_tokens_list: List[List[int]]) -> torch.Tensor:
        return self._forward_last_logits_full(prompt_tokens_list)
    
    @torch.no_grad()
    def _forward_stateful_active(self, step_tokens: List[List[int]], state, active_idx: List[int]) -> torch.Tensor:
        if not active_idx:
            return torch.empty((0, self.train_model.args.vocab_size), dtype=torch.float32, device=self.device)
        idx_tensor = torch.tensor(active_idx, device=self.device, dtype=torch.long)
        state0 = state[0].index_select(2, idx_tensor).contiguous()
        state1 = state[1].index_select(1, idx_tensor).contiguous()
        next_logits = self.infer_model.forward_batch(step_tokens, [state0, state1])
        if torch.is_tensor(next_logits) and next_logits.dim() == 3:
            next_logits = next_logits[:, -1, :]
        state[0][:, :, idx_tensor] = state0
        state[1][:, idx_tensor] = state1
        return next_logits

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
        stop_on_token_zero: bool = False,
        stop_on_repeat_ngram: bool = True,
        repeat_ngram_n: int = 16,
        repeat_ngram_repeat: int = 5,
        presence_penalty: float = 0.5,
        frequency_penalty: float = 0.1,
        alpha_decay: float = 0.99,
        post_trunc_append: str = " so the answer is : ",
        post_trunc_max_tokens: int = 0,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:
        """
        ?????????
        stateful ???? faster3a_2605???????? token ???????????????
        """
        Bp = len(prompt_tokens_list)
        if Bp == 0:
            return [], [], [], []

        if self.infer_model is not None:
            prep = getattr(self.infer_model, "prepare_stateful_rollout", None)
            if prep is not None:
                prep()

        last_logits, state = self.prime_prompts(prompt_tokens_list)

        B = Bp * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()
        full_rollout = self._use_full_rollout()

        if not full_rollout:
            state0 = state[0].repeat_interleave(group_size, dim=2).contiguous()
            state1 = state[1].repeat_interleave(group_size, dim=1).contiguous()
            state = [state0, state1]
            full_sequences = None
        else:
            full_sequences = []
            for prompt_tokens in prompt_tokens_list:
                for _ in range(group_size):
                    full_sequences.append(list(prompt_tokens))

        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        log_probs: List[List[float]] = [[] for _ in range(B)]
        comp_texts: List[str] = ["" for _ in range(B)]
        decoded_upto: List[int] = [0 for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]
        token_counts = torch.zeros((B, last_logits.size(-1)), device=last_logits.device)

        text_check_interval = 8

        for t in range(max_new_tokens):
            active_idx = active.nonzero(as_tuple=False).view(-1)
            if active_idx.numel() == 0:
                break

            cur_logits = last_logits.index_select(0, active_idx)
            cur_counts = token_counts.index_select(0, active_idx)
            token_ids, picked_logp = sample_next_token(
                logits=cur_logits,
                token_counts=cur_counts,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            cur_counts *= alpha_decay
            cur_counts.scatter_add_(1, token_ids.view(-1, 1), torch.ones_like(token_ids.view(-1, 1), dtype=torch.float32))
            token_counts.index_copy_(0, active_idx, cur_counts)

            idx_cpu = active_idx.detach().cpu().tolist()
            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()

            next_rows: List[int] = []
            step_tokens_batch: List[List[int]] = []
            for row, tok, lp in zip(idx_cpu, tok_cpu, lp_cpu):
                comp_tokens[row].append(int(tok))
                log_probs[row].append(float(lp))

                if stop_on_token_zero and int(tok) == 0:
                    del comp_tokens[row][-1:]
                    del log_probs[row][-1:]
                    active[row] = False
                    continue

                if stop_on_user and self._stop_token_id_seqs:
                    matched = self._match_stop_suffix_len(comp_tokens[row])
                    if matched > 0:
                        del comp_tokens[row][-matched:]
                        del log_probs[row][-matched:]
                        active[row] = False
                        continue

                pending = self.decode(comp_tokens[row][decoded_upto[row]:])
                if "\ufffd" not in pending:
                    comp_texts[row] += pending
                    decoded_upto[row] = len(comp_tokens[row])

                text = comp_texts[row]
                if stop_on_user and (not self._stop_token_id_seqs) and any(tok_s in text for tok_s in STOP_TOKENS):
                    active[row] = False
                    continue

                if (t < 2) or (len(comp_tokens[row]) % text_check_interval == 0):
                    if stop_on_think_close and "</think>" in text:
                        active[row] = False
                        continue
                    if stop_on_boxed and self._check_boxed_complete(text):
                        active[row] = False
                        continue
                    if stop_on_repeat_ngram and self._has_repeated_ngrams(text, n=repeat_ngram_n, repeat=repeat_ngram_repeat):
                        active[row] = False
                        continue

                next_rows.append(row)
                step_tokens_batch.append([int(tok)])

            if not next_rows:
                continue

            if full_rollout:
                for row, tok in zip(next_rows, [x[0] for x in step_tokens_batch]):
                    full_sequences[row].append(int(tok))
                new_logits = self._forward_last_logits_full([full_sequences[row] for row in next_rows])
            else:
                new_logits = self._forward_stateful_active(step_tokens_batch, state, next_rows)

            next_idx = torch.tensor(next_rows, device=last_logits.device, dtype=torch.long)
            last_logits.index_copy_(0, next_idx, new_logits)

        for i in range(B):
            if active[i]:
                truncated[i] = True
            if decoded_upto[i] < len(comp_tokens[i]):
                pending = self.decode(comp_tokens[i][decoded_upto[i]:])
                if "\ufffd" not in pending:
                    comp_texts[i] += pending
                    decoded_upto[i] = len(comp_tokens[i])

        return comp_tokens, log_probs, comp_texts, truncated

    def _match_stop_suffix_len(self, token_ids: List[int]) -> int:
        if not token_ids or not self._stop_token_id_seqs:
            return 0
        for seq in self._stop_token_id_seqs:
            n = len(seq)
            if n <= len(token_ids) and token_ids[-n:] == seq:
                return n
        return 0

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
