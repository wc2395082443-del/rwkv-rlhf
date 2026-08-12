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
        return getattr(self.cfg, "tune_mode", "state") == "full"
    
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

    @staticmethod
    def _compact_recurrent_state_in_place(
        state: List[torch.Tensor], keep_positions: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compact the rollout state while limiting temporary memory to one layer."""
        new_batch = keep_positions.numel()
        compact_state0 = state[0].index_select(2, keep_positions).contiguous()

        # state1 is roughly 6.4 GB at batch 512. Gathering it all at once can
        # OOM, so move each layer into the surviving prefix independently.
        state1 = state[1]
        for layer in range(state1.size(0)):
            gathered = state1[layer].index_select(0, keep_positions)
            state1[layer, :new_batch].copy_(gathered)
        compact_state1 = state1[:, :new_batch]
        return [compact_state0, compact_state1]

    @torch.no_grad()
    def _generate_group_parallel_gpu_buffered(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool], List[bool]]:
        """Generate the training fast path without per-token GPU-to-CPU copies."""
        last_logits, state = self.prime_prompts(prompt_tokens_list)
        batch_size = len(prompt_tokens_list) * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()
        state = [
            state[0].repeat_interleave(group_size, dim=2).contiguous(),
            state[1].repeat_interleave(group_size, dim=1).contiguous(),
        ]

        device = last_logits.device
        token_buffer = torch.zeros(
            (max_new_tokens, batch_size), dtype=torch.long, device=device
        )
        logp_buffer = torch.zeros(
            (max_new_tokens, batch_size), dtype=torch.float32, device=device
        )
        lengths = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        active_original_indices = torch.arange(batch_size, device=device)
        active_local = torch.ones((batch_size,), dtype=torch.bool, device=device)

        # Repetition penalties are disabled in sample_next_token, so no B x vocab
        # token-count tensor is needed on this training path.
        unused_token_counts = torch.empty((0,), dtype=torch.float32, device=device)
        compaction_interval = 16
        compaction_keep_ratio = 0.875
        forward_slots = 0
        compactions = 0
        generated_steps = 0
        for t in range(max_new_tokens):
            token_ids, picked_logp = sample_next_token(
                logits=last_logits,
                token_counts=unused_token_counts,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            was_active = active_local
            token_ids = torch.where(was_active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(was_active, picked_logp, torch.zeros_like(picked_logp))

            token_buffer[t].index_copy_(0, active_original_indices, token_ids)
            logp_buffer[t].index_copy_(0, active_original_indices, picked_logp)
            lengths.index_add_(
                0, active_original_indices, was_active.to(dtype=torch.int32)
            )
            just_ended = was_active & token_ids.eq(0)
            active_local = was_active & ~just_ended
            generated_steps = t + 1

            if generated_steps >= max_new_tokens:
                break

            should_check = generated_steps % compaction_interval == 0
            if should_check:
                survivor_count = int(active_local.sum().item())
                if survivor_count == 0:
                    break
                current_batch = active_original_indices.numel()
                if survivor_count <= int(current_batch * compaction_keep_ratio):
                    keep_positions = torch.nonzero(
                        active_local, as_tuple=False
                    ).squeeze(1)
                    survivor_tokens = token_ids.index_select(0, keep_positions)
                    active_original_indices = active_original_indices.index_select(
                        0, keep_positions
                    )
                    state = self._compact_recurrent_state_in_place(
                        state, keep_positions
                    )
                    active_local = torch.ones(
                        (survivor_count,), dtype=torch.bool, device=device
                    )
                    compactions += 1

                    forward_tokens_gpu = getattr(
                        self.infer_model, "forward_tokens_gpu", None
                    )
                    if forward_tokens_gpu is not None:
                        last_logits = forward_tokens_gpu(survivor_tokens, state)
                    else:
                        last_logits = self.infer_model.forward_batch(
                            survivor_tokens.unsqueeze(1), state
                        )
                    forward_slots += survivor_count
                    if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                        last_logits = last_logits[:, -1, :]
                    continue

            forward_tokens_gpu = getattr(self.infer_model, "forward_tokens_gpu", None)
            if forward_tokens_gpu is not None:
                last_logits = forward_tokens_gpu(token_ids, state)
            else:
                last_logits = self.infer_model.forward_batch(
                    token_ids.unsqueeze(1), state
                )
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]
            forward_slots += active_original_indices.numel()

        # One synchronization and host transfer replaces two transfers plus a
        # Python loop for every generated token.
        host_tokens = token_buffer[:generated_steps].transpose(0, 1).cpu()
        host_logps = logp_buffer[:generated_steps].transpose(0, 1).cpu()
        host_lengths = lengths.cpu().tolist()
        ended_eod_list = [
            bool(length and host_tokens[i, length - 1].item() == 0)
            for i, length in enumerate(host_lengths)
        ]
        truncated = [
            length >= max_new_tokens and not did_end_eod
            for length, did_end_eod in zip(host_lengths, ended_eod_list)
        ]

        baseline_slots = batch_size * max(0, generated_steps - 1)
        avoided_slots = max(0, baseline_slots - forward_slots)
        avoided_ratio = avoided_slots / max(1, baseline_slots)
        print(
            f"[rollout-compaction] count={compactions} "
            f"forward_slots={forward_slots}/{baseline_slots} "
            f"avoided={avoided_ratio:.3%} final_batch={active_original_indices.numel()}"
        )

        comp_tokens = [
            host_tokens[i, :length].tolist()
            for i, length in enumerate(host_lengths)
        ]
        log_probs = [
            host_logps[i, :length].tolist()
            for i, length in enumerate(host_lengths)
        ]
        comp_texts = []
        for tokens, did_end_eod in zip(comp_tokens, ended_eod_list):
            # Keep EOD in the policy trajectory, but match the original text
            # semantics by excluding it before decoding and reward checking.
            text_tokens = tokens[:-1] if did_end_eod and tokens and tokens[-1] == 0 else tokens
            comp_texts.append(self.decode(text_tokens))
        return comp_tokens, log_probs, comp_texts, truncated, ended_eod_list
    
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
        stop_on_token_zero: bool = True,
        stop_on_repeat_ngram: bool = True,
        repeat_ngram_n: int = 16,
        repeat_ngram_repeat: int = 5,
        presence_penalty: float = 0.5,
        frequency_penalty: float = 0.1,
        alpha_decay: float = 0.99,
        post_trunc_append: str = "",
        post_trunc_max_tokens: int = 0,
        post_trunc_close_think: bool = False,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool], List[bool]]:
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
            (comp_tokens, log_probs, comp_texts, truncated, ended_eod)
        """
        Bp = len(prompt_tokens_list)
        if Bp == 0:
            return [], [], [], [], []

        gpu_buffered_fast_path = (
            not self._use_full_rollout()
            and stop_on_token_zero
            and not stop_on_think_close
            and not stop_on_user
            and not stop_on_boxed
            and not stop_on_repeat_ngram
            and not post_trunc_append
            and post_trunc_max_tokens <= 0
        )
        if gpu_buffered_fast_path:
            return self._generate_group_parallel_gpu_buffered(
                prompt_tokens_list=prompt_tokens_list,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        
        # 1. 处理提示
        last_logits, state = self.prime_prompts(prompt_tokens_list)

        # 2. 复制状态以支持group_size
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
        
        # 3. 初始化生成状态
        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        log_probs: List[List[float]] = [[] for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]
        ended_eod = [False for _ in range(B)]
        
        # token计数器 (用于重复惩罚)
        token_counts = torch.zeros((B, last_logits.size(-1)), device=last_logits.device)
        
        # 4. ????
        for t in range(max_new_tokens):
            if not active.any():
                break

            # ?????token
            token_ids, picked_logp = sample_next_token(
                logits=last_logits,
                token_counts=token_counts,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            # ??token??
            token_counts.scatter_add_(
                1,
                token_ids.view(-1, 1),
                torch.ones_like(token_ids.view(-1, 1), dtype=torch.float32)
            )
            # ??????
            token_counts *= alpha_decay

            # ????
            token_ids = torch.where(active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(active, picked_logp, torch.zeros_like(picked_logp))

            # ????
            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()

            for i in range(B):
                if not active[i]:
                    continue
                # Keep EOD in the policy trajectory so Format-B reward can train the stop action.
                if stop_on_token_zero and int(tok_cpu[i]) == 0:
                    comp_tokens[i].append(0)
                    log_probs[i].append(float(lp_cpu[i]))
                    ended_eod[i] = True
                    active[i] = False
                    continue
                comp_tokens[i].append(int(tok_cpu[i]))
                log_probs[i].append(float(lp_cpu[i]))

            # ?????? (??)
            if stop_on_user and self._stop_token_id_seqs:
                for i in range(B):
                    if not active[i]:
                        continue
                    matched = self._match_stop_suffix_len(comp_tokens[i])
                    if matched > 0:
                        del comp_tokens[i][-matched:]
                        del log_probs[i][-matched:]
                        active[i] = False

            # Text-level checks are relatively expensive, keep sparse.
            if t % 10 == 0:
                for i in range(B):
                    if not active[i]:
                        continue

                    text = self.decode(comp_tokens[i])

                    if stop_on_think_close and '</think>' in text:
                        active[i] = False
                    elif stop_on_boxed and self._check_boxed_complete(text):
                        active[i] = False
                    elif stop_on_repeat_ngram and self._has_repeated_ngrams(text, n=repeat_ngram_n, repeat=repeat_ngram_repeat):
                        active[i] = False
                    elif stop_on_user and (not self._stop_token_id_seqs) and any(tok in text for tok in STOP_TOKENS):
                        active[i] = False

            if full_rollout:
                for i in range(B):
                    full_sequences[i].append(int(tok_cpu[i]))
                last_logits = self._forward_last_logits_full(full_sequences)
            else:
                step_tokens_batch = [[int(x)] for x in tok_cpu]
                last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
                if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                    last_logits = last_logits[:, -1, :]

        # 5. 标记截断的序列
        for i in range(B):
            if active[i]:
                truncated[i] = True

        # Evaluation may request a forced answer tail. Training rollouts pass an empty tail.
        if any(truncated) and post_trunc_append and post_trunc_max_tokens > 0:
            prefixes = [[] for _ in range(B)]
            for i in range(B):
                if not truncated[i]:
                    continue
                prefix = post_trunc_append
                if post_trunc_close_think and "</think>" not in self.decode(comp_tokens[i]):
                    prefix = "\n</think>\n" + prefix
                prefixes[i] = [int(tok) for tok in self.encode(prefix)]

            # Prefix lengths can differ when a response already closed </think>.
            # Leading newlines keep recurrent updates batch-aligned without changing meaning.
            max_prefix_len = max((len(ids) for ids in prefixes), default=0)
            newline_ids = [int(tok) for tok in self.encode("\n")]
            if len(newline_ids) != 1:
                raise RuntimeError("expected newline to encode to one token")
            newline_id = newline_ids[0]
            for i in range(B):
                if truncated[i] and len(prefixes[i]) < max_prefix_len:
                    prefixes[i] = [newline_id] * (max_prefix_len - len(prefixes[i])) + prefixes[i]

            for pos in range(max_prefix_len):
                step_tokens = [prefixes[i][pos] if truncated[i] else 0 for i in range(B)]
                token_ids = torch.tensor(step_tokens, device=last_logits.device, dtype=torch.long)
                logp_all = F.log_softmax(last_logits, dim=-1)
                picked_logp = logp_all.gather(1, token_ids.view(-1, 1)).squeeze(1)
                token_counts.scatter_add_(
                    1,
                    token_ids.view(-1, 1),
                    torch.ones_like(token_ids.view(-1, 1), dtype=torch.float32),
                )
                token_counts *= alpha_decay

                lp_cpu = picked_logp.detach().cpu().tolist()
                for i in range(B):
                    if truncated[i]:
                        comp_tokens[i].append(int(step_tokens[i]))
                        log_probs[i].append(float(lp_cpu[i]))

                if full_rollout:
                    for i in range(B):
                        if truncated[i]:
                            full_sequences[i].append(int(step_tokens[i]))
                    last_logits = self._forward_last_logits_full(full_sequences)
                else:
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
                    if not active_tail[i]:
                        continue
                    if stop_on_token_zero and int(tok_cpu[i]) == 0:
                        comp_tokens[i].append(0)
                        log_probs[i].append(float(lp_cpu[i]))
                        ended_eod[i] = True
                        active_tail[i] = False
                        continue
                    comp_tokens[i].append(int(tok_cpu[i]))
                    log_probs[i].append(float(lp_cpu[i]))

                if stop_on_user and self._stop_token_id_seqs:
                    for i in range(B):
                        if not active_tail[i]:
                            continue
                        matched = self._match_stop_suffix_len(comp_tokens[i])
                        if matched > 0:
                            del comp_tokens[i][-matched:]
                            del log_probs[i][-matched:]
                            active_tail[i] = False

                if t % 10 == 0:
                    for i in range(B):
                        if not active_tail[i]:
                            continue

                        text = self.decode(comp_tokens[i])

                        if stop_on_think_close and '</think>' in text:
                            active_tail[i] = False
                        elif stop_on_boxed and self._check_boxed_complete(text):
                            active_tail[i] = False
                        elif stop_on_repeat_ngram and self._has_repeated_ngrams(text, n=repeat_ngram_n, repeat=repeat_ngram_repeat):
                            active_tail[i] = False
                        elif stop_on_user and (not self._stop_token_id_seqs) and any(tok in text for tok in STOP_TOKENS):
                            active_tail[i] = False

                if full_rollout:
                    for i in range(B):
                        full_sequences[i].append(int(tok_cpu[i]))
                    last_logits = self._forward_last_logits_full(full_sequences)
                else:
                    step_tokens_batch = [[int(x)] for x in tok_cpu]
                    last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
                    if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                        last_logits = last_logits[:, -1, :]

        # 7. 解码文本
        comp_texts = []
        for i in range(B):
            text_tokens = comp_tokens[i][:-1] if ended_eod[i] and comp_tokens[i] and comp_tokens[i][-1] == 0 else comp_tokens[i]
            text = self.decode(text_tokens)
            comp_texts.append(text)

        return comp_tokens, log_probs, comp_texts, truncated, ended_eod
    

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
