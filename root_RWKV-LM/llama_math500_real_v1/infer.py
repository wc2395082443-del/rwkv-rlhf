#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
import torch
import torch.nn.functional as F

STOP_TOKENS = [
    "<|start_header_id|>user<|end_header_id|>",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|start_header_id|>system<|end_header_id|>",
    "<|start_header_id|>ipython<|end_header_id|>",
    "<|begin_of_text|>",
]


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0 or temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    logits = logits.clone()
    logits[logits < v[:, [-1]]] = float('-inf')
    return logits


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs = probs.clone()
    probs[indices_to_remove] = 0.0
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
    return probs


class HFCausalBatchInference:
    def __init__(self, infer_model, train_model, tokenizer, encode_fn, decode_fn, device: str, cfg):
        self.infer_model = infer_model
        self.train_model = train_model
        self.tokenizer = tokenizer
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        self.cfg = cfg
        self.pad_token_id = int(cfg.pad_token_id)
        self.eos_token_id = int(cfg.eos_token_id)
        self.gen_batch_size = max(1, int(getattr(cfg, 'gen_batch_size', 8)))
        self._stop_token_id_seqs: List[List[int]] = []
        for s in STOP_TOKENS:
            try:
                ids = self.encode(s)
            except Exception:
                ids = None
            if ids:
                self._stop_token_id_seqs.append([int(x) for x in ids])
        self._stop_token_id_seqs.sort(key=len, reverse=True)

    def _pad_left(self, seqs: List[List[int]]):
        max_len = max(len(s) for s in seqs)
        padded = []
        masks = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append([self.pad_token_id] * pad_len + list(s))
            masks.append([0] * pad_len + [1] * len(s))
        return (
            torch.tensor(padded, device=self.device, dtype=torch.long),
            torch.tensor(masks, device=self.device, dtype=torch.long),
        )

    def _find_stop_pos(self, tokens: List[int]) -> Tuple[int, bool]:
        eos_pos = None
        for i, tok in enumerate(tokens):
            if int(tok) == self.eos_token_id:
                eos_pos = i
                break
        stop_pos = None
        for seq in self._stop_token_id_seqs:
            n = len(seq)
            if n == 0 or len(tokens) < n:
                continue
            for i in range(0, len(tokens) - n + 1):
                if tokens[i:i+n] == seq:
                    stop_pos = i if stop_pos is None else min(stop_pos, i)
                    break
        if stop_pos is not None and (eos_pos is None or stop_pos <= eos_pos):
            return stop_pos, True
        if eos_pos is not None:
            return eos_pos, False
        return len(tokens), False

    def _hit_stop_suffix(self, tokens: List[int]) -> bool:
        if not tokens:
            return False
        if int(tokens[-1]) == self.eos_token_id:
            return True
        for seq in self._stop_token_id_seqs:
            n = len(seq)
            if n > 0 and len(tokens) >= n and tokens[-n:] == seq:
                return True
        return False

    @torch.no_grad()
    def _sample_chunk_online_logps(
        self,
        prompt_tokens_list: List[List[int]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        input_ids, attention_mask = self._pad_left(prompt_tokens_list)
        batch_size = input_ids.size(0)
        generated_tokens = [[] for _ in range(batch_size)]
        old_logps_list = [[] for _ in range(batch_size)]
        active = torch.ones(batch_size, device=self.device, dtype=torch.bool)

        self.train_model.eval()
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        past_key_values = None

        for _ in range(max_new_tokens):
            outputs = self.train_model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

            raw_logits = next_logits.float()
            sample_logits = apply_temperature(raw_logits, temperature)
            sample_logits = apply_top_k(sample_logits, top_k)
            probs = F.softmax(sample_logits, dim=-1)
            probs = apply_top_p(probs, top_p)
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if not bool(active.all().item()):
                token_ids = torch.where(active, token_ids, torch.full_like(token_ids, self.eos_token_id))

            raw_logsumexp = torch.logsumexp(raw_logits, dim=-1)
            raw_target_logits = raw_logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            token_logps = raw_target_logits - raw_logsumexp
            if not bool(active.all().item()):
                token_logps = torch.where(active, token_logps, torch.zeros_like(token_logps))

            active_list = active.detach().cpu().tolist()
            token_ids_list = token_ids.detach().cpu().tolist()
            token_logps_vals = token_logps.detach().cpu().tolist()

            next_active = active.clone()
            any_active = False
            for i, is_active in enumerate(active_list):
                if not is_active:
                    continue
                generated_tokens[i].append(int(token_ids_list[i]))
                old_logps_list[i].append(float(token_logps_vals[i]))
                if self._hit_stop_suffix(generated_tokens[i]):
                    next_active[i] = False
                else:
                    any_active = True

            active = next_active
            if not any_active:
                break

            cur_input_ids = token_ids.unsqueeze(-1)
            step_mask = torch.ones((batch_size, 1), device=self.device, dtype=cur_attention_mask.dtype)
            cur_attention_mask = torch.cat([cur_attention_mask, step_mask], dim=1)

            del outputs, next_logits, raw_logits, sample_logits, probs, token_ids
            del raw_logsumexp, raw_target_logits, token_logps, step_mask

        comp_tokens_list = []
        comp_texts_list = []
        truncated_list = []
        trimmed_logps_list = []
        for generated, logps in zip(generated_tokens, old_logps_list):
            cut, by_stop = self._find_stop_pos(generated)
            comp_tokens = generated[:cut]
            comp_text = self.tokenizer.decode(comp_tokens, skip_special_tokens=True)
            truncated = bool(by_stop or (cut >= max_new_tokens and len(generated) >= max_new_tokens))
            comp_tokens_list.append(comp_tokens)
            comp_texts_list.append(comp_text)
            truncated_list.append(truncated)
            trimmed_logps_list.append(logps[:cut])

        return comp_tokens_list, trimmed_logps_list, comp_texts_list, truncated_list

    @torch.no_grad()
    def _sample_chunk(
        self,
        prompt_tokens_list: List[List[int]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        return_logps: bool = True,
    ):
        if return_logps:
            return self._sample_chunk_online_logps(
                prompt_tokens_list=prompt_tokens_list,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        input_ids, attention_mask = self._pad_left(prompt_tokens_list)
        self.train_model.eval()
        outputs = self.train_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=max(float(temperature), 1e-5),
            top_p=float(top_p) if 0.0 < float(top_p) < 1.0 else 1.0,
            top_k=max(0, int(top_k)),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False,
        )
        prompt_pad_len = input_ids.size(1)
        comp_tokens_list = []
        comp_texts_list = []
        truncated_list = []
        for i, _prompt_tokens in enumerate(prompt_tokens_list):
            generated = outputs[i, prompt_pad_len:].tolist()
            cut, by_stop = self._find_stop_pos(generated)
            comp_tokens = generated[:cut]
            comp_text = self.tokenizer.decode(comp_tokens, skip_special_tokens=True)
            truncated = bool(by_stop or (cut >= max_new_tokens and len(generated) >= max_new_tokens))
            comp_tokens_list.append(comp_tokens)
            comp_texts_list.append(comp_text)
            truncated_list.append(truncated)

        old_logps_list = [[] for _ in prompt_tokens_list]
        return comp_tokens_list, old_logps_list, comp_texts_list, truncated_list

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
        post_trunc_append: str = ' so the answer is : ',
        post_trunc_max_tokens: int = 64,
        gen_batch_size: int = None,
        return_logps: bool = True,
    ):
        if not prompt_tokens_list:
            return [], [], [], []
        expanded = []
        for p in prompt_tokens_list:
            for _ in range(group_size):
                expanded.append(list(p))

        all_comp_tokens = []
        all_old_logps = []
        all_texts = []
        all_truncated = []
        batch_size = max(1, int(gen_batch_size)) if gen_batch_size is not None else self.gen_batch_size
        for start in range(0, len(expanded), batch_size):
            chunk = expanded[start:start + batch_size]
            ct, lp, txt, trunc = self._sample_chunk(
                prompt_tokens_list=chunk,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                return_logps=return_logps,
            )
            all_comp_tokens.extend(ct)
            all_old_logps.extend(lp)
            all_texts.extend(txt)
            all_truncated.extend(trunc)
        return all_comp_tokens, all_old_logps, all_texts, all_truncated
