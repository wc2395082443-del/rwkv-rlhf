#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import copy
import sys
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import read_jsonl, set_seed, build_prompt, build_chat_messages
from infer import HFCausalBatchInference
from train import GRPOTrainer, GRPOConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_jsonl', type=str, required=True)
    ap.add_argument('--eval_jsonl', type=str, default=None)
    ap.add_argument('--max_data_samples', type=int, default=None)

    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--ctx_len', type=int, default=2048)

    ap.add_argument('--num_questions', type=int, default=24)
    ap.add_argument('--samples_per_question', type=int, default=8)
    ap.add_argument('--hard_buffer_ttl', type=int, default=4)
    ap.add_argument('--hard_buffer_cooldown', type=int, default=4)
    ap.add_argument('--hard_buffer_target_samples', type=int, default=192)
    ap.add_argument('--hard_buffer_group_size', type=int, default=8)
    ap.add_argument('--hard_buffer_extra_lr_scale', type=float, default=0.5)
    ap.add_argument('--hard_buffer_adv_clip', type=float, default=2.5)
    ap.add_argument('--dynamic_resample_enable', type=int, default=1)
    ap.add_argument('--dynamic_min_effective_groups', type=int, default=6)
    ap.add_argument('--dynamic_max_rounds', type=int, default=3)
    ap.add_argument('--dynamic_unique_questions', type=int, default=1)

    ap.add_argument('--max_new_tokens', type=int, default=768)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.6)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--eval_temperature', type=float, default=0.3)
    ap.add_argument('--eval_top_p', type=float, default=0.4)
    ap.add_argument('--eval_top_k', type=int, default=500)
    ap.add_argument('--gen_batch_size', type=int, default=8)
    ap.add_argument('--eval_gen_batch_size', type=int, default=None)
    ap.add_argument('--pre_eval_gen_batch_size', type=int, default=None)

    ap.add_argument('--min_tokens', type=int, default=200)
    ap.add_argument('--length_weight', type=float, default=0.0)
    ap.add_argument('--zstd_threshold', type=float, default=2.5)
    ap.add_argument('--zstd_penalty_weight', type=float, default=0.2)
    ap.add_argument('--ngram_penalty', type=float, default=0.0)
    ap.add_argument('--answer_judge', type=str, default='legacy', choices=['legacy', 'math_verify', 'auto'])

    ap.add_argument('--total_steps', type=int, default=100)
    ap.add_argument('--ppo_epochs', type=int, default=1)
    ap.add_argument('--micro_batch', type=int, default=1)
    ap.add_argument('--lr', type=float, default=6e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--kl_coef', type=float, default=0.05)
    ap.add_argument('--kl_mode', type=str, default='k3_loss', choices=['k1_reward', 'k3_loss'])
    ap.add_argument('--neg_adv_weight', type=float, default=0.6)

    ap.add_argument('--time_state_l2', type=float, default=0.0)
    ap.add_argument('--time_state_clamp', type=float, default=0.0)

    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.0)

    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--log_interval', type=int, default=1)
    ap.add_argument('--save_interval', type=int, default=50)
    ap.add_argument('--eval_interval', type=int, default=5)
    ap.add_argument('--eval_sample_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip_pre_eval', action='store_true')
    return ap.parse_args()


def freeze_non_lora(model: torch.nn.Module) -> int:
    trainable = 0
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
    return trainable


def load_models(model_path: str, device: str, ctx_len: int, lora_r: int, lora_alpha: int, lora_dropout: float):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError('Tokenizer has no pad_token_id')

    common_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    train_base = AutoModelForCausalLM.from_pretrained(**common_kwargs).to(device)
    train_base.config.use_cache = False
    train_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=target_modules,
    )
    train_model = get_peft_model(train_base, lora_cfg)
    train_model.enable_input_require_grads()
    train_model.args = SimpleNamespace(ctx_len=ctx_len)

    ref_model = AutoModelForCausalLM.from_pretrained(**common_kwargs).to(device)
    ref_model.eval()
    ref_model.config.use_cache = False
    ref_model.args = SimpleNamespace(ctx_len=ctx_len)
    for p in ref_model.parameters():
        p.requires_grad = False

    trainable = freeze_non_lora(train_model)
    return tokenizer, train_model, ref_model, trainable


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    train_data = read_jsonl(args.train_jsonl, max_samples=args.max_data_samples)
    if not train_data:
        raise RuntimeError('??????')
    if args.eval_jsonl:
        test_data = read_jsonl(args.eval_jsonl)
    else:
        test_size = min(128, len(train_data) // 5)
        test_data = train_data[:test_size]
        train_data = train_data[test_size:]

    print(f'????: {args.model}')
    tokenizer, train_model, ref_model, trainable = load_models(
        model_path=args.model,
        device=device,
        ctx_len=int(args.ctx_len),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    print(f'?????: {trainable}')

    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)

    use_chat_template = bool(getattr(tokenizer, 'chat_template', None))
    if use_chat_template:
        def prompt_encoder(problem: str):
            rendered = tokenizer.apply_chat_template(
                build_chat_messages(problem),
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(rendered, 'input_ids'):
                return rendered.input_ids
            return rendered
    else:
        def prompt_encoder(problem: str):
            return tokenizer.encode(build_prompt(problem), add_special_tokens=False)

    cfg = GRPOConfig(
        num_questions=int(args.num_questions),
        samples_per_question=int(args.samples_per_question),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        eval_temperature=float(args.eval_temperature),
        eval_top_p=float(args.eval_top_p),
        eval_top_k=int(args.eval_top_k),
        ppo_epochs=int(args.ppo_epochs),
        micro_batch=int(args.micro_batch),
        lr=float(args.lr),
        grad_clip=float(args.grad_clip),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_new_tokens),
        length_weight=float(args.length_weight),
        zstd_threshold=float(args.zstd_threshold),
        zstd_penalty_weight=float(args.zstd_penalty_weight),
        ngram_penalty=float(args.ngram_penalty),
        answer_judge=str(args.answer_judge),
        neg_adv_weight=float(args.neg_adv_weight),
        kl_coef=float(args.kl_coef),
        kl_mode=str(args.kl_mode),
        time_state_l2=float(args.time_state_l2),
        time_state_clamp=float(args.time_state_clamp),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        eval_interval=int(args.eval_interval),
        eval_sample_ratio=float(args.eval_sample_ratio),
        hard_buffer_ttl=int(args.hard_buffer_ttl),
        hard_buffer_cooldown=int(args.hard_buffer_cooldown),
        hard_buffer_target_samples=int(args.hard_buffer_target_samples),
        hard_buffer_group_size=int(args.hard_buffer_group_size),
        hard_buffer_extra_lr_scale=float(args.hard_buffer_extra_lr_scale),
        hard_buffer_adv_clip=float(args.hard_buffer_adv_clip),
        dynamic_resample_enable=bool(int(args.dynamic_resample_enable)),
        dynamic_min_effective_groups=int(args.dynamic_min_effective_groups),
        dynamic_max_rounds=int(args.dynamic_max_rounds),
        dynamic_unique_questions=bool(int(args.dynamic_unique_questions)),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        gen_batch_size=int(args.gen_batch_size),
        eval_gen_batch_size=(int(args.eval_gen_batch_size) if args.eval_gen_batch_size is not None else int(args.gen_batch_size)),
        pre_eval_gen_batch_size=(int(args.pre_eval_gen_batch_size) if args.pre_eval_gen_batch_size is not None else ((int(args.eval_gen_batch_size) if args.eval_gen_batch_size is not None else int(args.gen_batch_size)))),
    )

    infer_engine = HFCausalBatchInference(
        infer_model=train_model,
        train_model=train_model,
        tokenizer=tokenizer,
        encode_fn=encode,
        decode_fn=decode,
        device=device,
        cfg=cfg,
    )

    print(f'prompt_mode={"chat_template" if use_chat_template else "plain"}')

    trainer = GRPOTrainer(
        train_model=train_model,
        ref_model=ref_model,
        infer_engine=infer_engine,
        encode_fn=encode,
        decode_fn=decode,
        prompt_encoder_fn=prompt_encoder,
        train_data=train_data,
        test_data=test_data,
        out_dir=args.out_dir,
        device=device,
        cfg=cfg,
        seed=int(args.seed),
    )

    print('\n' + '=' * 60)
    print(f'Llama GRPO baseline | total_steps={args.total_steps} | device={device}')
    print(f'model={args.model}')
    print(f'data={args.train_jsonl}')
    print('=' * 60 + '\n')

    if not args.skip_pre_eval:
        pre_acc = trainer.evaluate(step=0, tag='pre_eval', sample_ratio=1.0)
        if pre_acc is not None:
            print(f'[pre_eval] acc={pre_acc:.4f}')
    else:
        print('[pre_eval] skipped')
    trainer.train(total_steps=int(args.total_steps))
    post_acc = trainer.evaluate(step=int(args.total_steps), tag='post_eval')
    if post_acc is not None:
        print(f'[post_eval] acc={post_acc:.4f}')


if __name__ == '__main__':
    main()

