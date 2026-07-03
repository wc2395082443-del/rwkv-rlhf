from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import dataset_root, resolve_model_path, table
from .env import env_value, pick
from .paths import resolve_path


WKV_MODES = ("fp16", "fp32io16")
EMB_DEVICES = ("cpu", "gpu")


@dataclass
class CommandPlan:
    command: list[str]
    cwd: Path
    shown_env: dict[str, str]
    env: dict[str, str]


def format_hydra_file_list(value: Any, *, root: Path, env: dict[str, str]) -> str:
    if isinstance(value, list):
        files = [str(resolve_path(str(path), root=root, env=env)) for path in value]
        return "[" + ",".join(f"'{path}'" for path in files) + "]"
    return str(value)


def format_hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def prepend_venv_path(env: dict[str, str], root: Path, config: dict[str, Any]) -> None:
    paths = table(config, "paths")
    venv_value = pick(
        paths.get("venv"),
        env_value(env, "HELICOPTER_VENV", "VENV", "REMOTE_VENV"),
    )
    if not venv_value:
        venv_value = ".venv"
    venv = resolve_path(str(venv_value), root=root, env=env)
    bin_dir = venv / "bin"
    if bin_dir.exists():
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"


def python_executable(
    config: dict[str, Any],
    *,
    root: Path,
    env: dict[str, str],
    require_configured: bool = False,
) -> str:
    paths = table(config, "paths")
    python_value = pick(paths.get("python"), env_value(env, "HELICOPTER_PYTHON", "PYTHON"))
    if python_value:
        python = resolve_path(str(python_value), root=root, env=env)
        if require_configured and not os.access(python, os.X_OK):
            raise SystemExit(f"Python executable not found: {python}")
        return str(python)

    venv_value = pick(
        paths.get("venv"),
        env_value(env, "HELICOPTER_VENV", "VENV", "REMOTE_VENV"),
        ".venv",
    )
    venv = resolve_path(str(venv_value), root=root, env=env)
    python = venv / "bin/python"
    if python.exists():
        return str(python)
    if require_configured:
        raise SystemExit(
            f"Python executable not found: {python}; run scripts/install_local.sh "
            "or set HELICOPTER_PYTHON / paths.python"
        )
    return str(Path(sys.executable))


def apply_rwkv_env(
    command_env: dict[str, str],
    *,
    wkv_mode: str | None,
    emb_device: str | None,
) -> None:
    if wkv_mode is not None:
        command_env["VLLM_RWKV7_WKV_MODE"] = wkv_mode
    if emb_device is not None:
        command_env["VLLM_RWKV7_EMB_DEVICE"] = emb_device


def strip_vllm_env(env: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in env.items() if not key.startswith("VLLM_")}


def takeoff_value(
    takeoff: dict[str, Any],
    env: dict[str, str],
    config_key: str,
    env_key: str,
    default: Any = None,
) -> Any:
    return pick(env_value(env, env_key), takeoff.get(config_key), default)


def append_hydra_override(overrides: list[str], key: str, value: Any, *, optional: bool = False) -> None:
    if optional and (value is None or str(value) == ""):
        return
    overrides.append(f"{key}={format_hydra_value(value)}")


def build_grpo_hydra_overrides(
    *,
    model_path: Path,
    data_root: Path,
    dataset: dict[str, Any],
    takeoff: dict[str, Any],
    env: dict[str, str],
    root: Path,
    verl_path: Path,
    rwkv_lm_path: Path,
    num_nodes: Any,
    num_devices: Any,
) -> list[str]:
    train_files = env_value(env, "TRAIN_FILES")
    if train_files is None:
        if "train_files" in dataset:
            train_files = format_hydra_file_list(dataset["train_files"], root=root, env=env)
        else:
            train_files = f"['{data_root}/train.parquet']"

    val_files = env_value(env, "VAL_FILES")
    if val_files is None:
        if "val_files" in dataset:
            val_files = format_hydra_file_list(dataset["val_files"], root=root, env=env)
        else:
            val_files = f"['{data_root}/test.parquet']"

    dynamic_bsz = takeoff_value(takeoff, env, "rwkv_use_dynamic_bsz", "RWKV_USE_DYNAMIC_BSZ", False)
    ppo_micro_batch_size = takeoff_value(takeoff, env, "ppo_micro_batch_size", "PPO_MICRO_BATCH_SIZE", 8)
    ppo_max_token_len_per_gpu = takeoff_value(
        takeoff,
        env,
        "ppo_max_token_len_per_gpu",
        "PPO_MAX_TOKEN_LEN_PER_GPU",
        8192,
    )
    rollout_tensor_parallel_size = takeoff_value(
        takeoff,
        env,
        "rollout_tensor_parallel_size",
        "ROLLOUT_TP",
        1,
    )
    rollout_gpu_memory_utilization = takeoff_value(
        takeoff,
        env,
        "rollout_gpu_memory_utilization",
        "ROLLOUT_GPU_MEM_UTIL",
    )
    rollout_n = takeoff_value(takeoff, env, "rollout_n", "ROLLOUT_N", 8)
    rollout_max_num_seqs = takeoff_value(takeoff, env, "rollout_max_num_seqs", "ROLLOUT_MAX_NUM_SEQS")
    rollout_max_num_batched_tokens = takeoff_value(
        takeoff,
        env,
        "rollout_max_num_batched_tokens",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
    )
    rollout_n_gpus_per_node = takeoff_value(
        takeoff,
        env,
        "rollout_n_gpus_per_node",
        "ROLLOUT_NGPUS_PER_NODE",
    )
    rollout_data_parallel_size = takeoff_value(
        takeoff,
        env,
        "rollout_data_parallel_size",
        "ROLLOUT_DP",
    )
    rollout_pipeline_parallel_size = takeoff_value(
        takeoff,
        env,
        "rollout_pipeline_parallel_size",
        "ROLLOUT_PP",
    )
    trainer_n_gpus_per_node = takeoff_value(
        takeoff,
        env,
        "trainer_n_gpus_per_node",
        "TRAIN_NGPUS_PER_NODE",
        num_devices,
    )

    reward_path = verl_path / "examples/rwkv_trainer/math_dapo_reward.py"
    overrides = [
        f"algorithm.adv_estimator={format_hydra_value(takeoff_value(takeoff, env, 'adv_estimator', 'ADV_ESTIMATOR', 'grpo'))}",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_files}",
        f"data.val_files={val_files}",
        f"data.train_batch_size={format_hydra_value(takeoff_value(takeoff, env, 'train_batch_size', 'TRAIN_BATCH_SIZE', 56))}",
        f"data.max_prompt_length={format_hydra_value(takeoff_value(takeoff, env, 'max_prompt_length', 'MAX_PROMPT_LENGTH', 512))}",
        f"data.max_response_length={format_hydra_value(takeoff_value(takeoff, env, 'max_response_length', 'MAX_RESPONSE_LENGTH', 512))}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        f"reward.reward_manager.name={format_hydra_value(takeoff_value(takeoff, env, 'reward_manager', 'REWARD_MANAGER', 'naive'))}",
        "model@actor_rollout_ref.model=rwkv_native",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.model.rwkv_lm_path={rwkv_lm_path}",
        "actor@actor_rollout_ref.actor=rwkv_lm",
        f"actor_rollout_ref.actor.engine.rwkv_lm_path={rwkv_lm_path}",
        f"actor_rollout_ref.actor.optim.lr={format_hydra_value(takeoff_value(takeoff, env, 'actor_lr', 'ACTOR_LR', '1e-5'))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={format_hydra_value(takeoff_value(takeoff, env, 'ppo_mini_batch_size', 'PPO_MINI_BATCH_SIZE', 56))}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={format_hydra_value(ppo_micro_batch_size)}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={format_hydra_value(dynamic_bsz)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={format_hydra_value(ppo_max_token_len_per_gpu)}",
        f"actor_rollout_ref.actor.use_kl_loss={format_hydra_value(takeoff_value(takeoff, env, 'actor_use_kl_loss', 'ACTOR_USE_KL_LOSS', False))}",
        f"actor_rollout_ref.actor.kl_loss_coef={format_hydra_value(takeoff_value(takeoff, env, 'actor_kl_loss_coef', 'ACTOR_KL_LOSS_COEF', 0.0))}",
        f"actor_rollout_ref.actor.kl_loss_type={format_hydra_value(takeoff_value(takeoff, env, 'actor_kl_loss_type', 'ACTOR_KL_LOSS_TYPE', 'low_var_kl'))}",
    ]
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.optim.lr_warmup_steps",
        takeoff_value(takeoff, env, "actor_lr_warmup_steps", "ACTOR_LR_WARMUP_STEPS"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.optim.weight_decay",
        takeoff_value(takeoff, env, "actor_weight_decay", "ACTOR_WEIGHT_DECAY"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.entropy_coeff",
        takeoff_value(takeoff, env, "actor_entropy_coeff", "ACTOR_ENTROPY_COEFF"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.optim.clip_grad",
        takeoff_value(takeoff, env, "actor_grad_clip", "ACTOR_GRAD_CLIP"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.clip_ratio_low",
        takeoff_value(takeoff, env, "clip_ratio_low", "CLIP_RATIO_LOW"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.clip_ratio_high",
        takeoff_value(takeoff, env, "clip_ratio_high", "CLIP_RATIO_HIGH"),
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.actor.clip_ratio_c",
        takeoff_value(takeoff, env, "clip_ratio_c", "CLIP_RATIO_C"),
        optional=True,
    )

    overrides.extend(
        [
            "ref@actor_rollout_ref.ref=rwkv_lm",
            f"actor_rollout_ref.ref.engine.rwkv_lm_path={rwkv_lm_path}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={format_hydra_value(ppo_micro_batch_size)}",
            f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={format_hydra_value(dynamic_bsz)}",
            f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={format_hydra_value(ppo_max_token_len_per_gpu)}",
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.load_format=auto",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={format_hydra_value(rollout_tensor_parallel_size)}",
            f"actor_rollout_ref.rollout.n={format_hydra_value(rollout_n)}",
            "actor_rollout_ref.rollout.enable_prefix_caching=False",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={format_hydra_value(ppo_micro_batch_size)}",
            f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={format_hydra_value(dynamic_bsz)}",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={format_hydra_value(ppo_max_token_len_per_gpu)}",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=rwkv",
            "actor_rollout_ref.hybrid_engine=False",
            f"rollout.nnodes={format_hydra_value(num_nodes)}",
        ]
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        rollout_gpu_memory_utilization,
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.rollout.max_num_seqs",
        rollout_max_num_seqs,
        optional=True,
    )
    append_hydra_override(
        overrides,
        "actor_rollout_ref.rollout.max_num_batched_tokens",
        rollout_max_num_batched_tokens,
        optional=True,
    )
    append_hydra_override(overrides, "rollout.n_gpus_per_node", rollout_n_gpus_per_node, optional=True)
    for config_key, env_key, hydra_key in (
        (
            "rollout_n_gpus_per_node",
            "ROLLOUT_NGPUS_PER_NODE",
            "actor_rollout_ref.rollout.n_gpus_per_node",
        ),
        ("rollout_mode", "ROLLOUT_MODE", "actor_rollout_ref.rollout.mode"),
        ("rollout_data_parallel_size", "ROLLOUT_DP", "actor_rollout_ref.rollout.data_parallel_size"),
        (
            "rollout_pipeline_parallel_size",
            "ROLLOUT_PP",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size",
        ),
        (
            "rollout_checkpoint_engine_backend",
            "ROLLOUT_CHECKPOINT_ENGINE_BACKEND",
            "actor_rollout_ref.rollout.checkpoint_engine.backend",
        ),
        (
            "rollout_correction_bypass_mode",
            "ROLLOUT_CORRECTION_BYPASS_MODE",
            "algorithm.rollout_correction.bypass_mode",
        ),
    ):
        append_hydra_override(
            overrides,
            hydra_key,
            takeoff_value(takeoff, env, config_key, env_key),
            optional=True,
        )

    overrides.extend(
        [
            "critic.enable=False",
            'trainer.logger=["console"]',
            f"trainer.project_name={format_hydra_value(takeoff_value(takeoff, env, 'project_name', 'PROJECT_NAME', 'verl_rwkv_grpo'))}",
            f"trainer.experiment_name={format_hydra_value(takeoff_value(takeoff, env, 'experiment_name', 'EXPERIMENT_NAME', 'rwkv7_grpo_vllm'))}",
            f"trainer.nnodes={format_hydra_value(num_nodes)}",
            f"trainer.n_gpus_per_node={format_hydra_value(trainer_n_gpus_per_node)}",
            f"trainer.save_freq={format_hydra_value(takeoff_value(takeoff, env, 'save_freq', 'SAVE_FREQ', 20))}",
            f"trainer.test_freq={format_hydra_value(takeoff_value(takeoff, env, 'test_freq', 'TEST_FREQ', -1))}",
            f"trainer.val_before_train={format_hydra_value(takeoff_value(takeoff, env, 'val_before_train', 'VAL_BEFORE_TRAIN', False))}",
            f"trainer.total_epochs={format_hydra_value(takeoff_value(takeoff, env, 'total_epochs', 'TOTAL_EPOCHS', 2))}",
        ]
    )
    append_hydra_override(
        overrides,
        "trainer.total_training_steps",
        takeoff_value(takeoff, env, "total_training_steps", "TOTAL_TRAINING_STEPS"),
        optional=True,
    )
    return overrides


def build_infer_plan(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> CommandPlan:
    model_path, model = resolve_model_path(config, args.model, root=root, env=env)
    infer = table(config, "infer")
    runtime = table(config, "runtime")
    gpu = table(config, "gpu")

    wkv_mode_value = pick(
        args.wkv_mode,
        env_value(env, "HELICOPTER_INFER_WKV_MODE", "VLLM_RWKV7_WKV_MODE"),
        infer.get("wkv_mode"),
    )
    wkv_mode = str(wkv_mode_value) if wkv_mode_value is not None else None
    emb_device_value = pick(
        args.emb_device,
        env_value(env, "HELICOPTER_INFER_EMB_DEVICE", "VLLM_RWKV7_EMB_DEVICE"),
        infer.get("emb_device"),
    )
    emb_device = str(emb_device_value) if emb_device_value is not None else None
    host = str(pick(args.host, runtime.get("host"), default="0.0.0.0"))
    port = str(pick(args.port, runtime.get("port"), default="8000"))
    served_model_name = str(
        pick(args.served_model_name, model.get("served_model_name"), model.get("requested_name"), args.model)
    )

    if not args.dry_run and not model_path.is_file():
        raise SystemExit(f"RWKV checkpoint not found: {model_path}")

    command = [
        "vllm",
        "serve",
        str(model_path),
        "--host",
        host,
        "--port",
        port,
        "--tokenizer-mode",
        "rwkv",
        "--load-format",
        "auto",
        "--served-model-name",
        served_model_name,
    ]

    option_values = {
        "--tensor-parallel-size": pick(
            args.tensor_parallel_size,
            env_value(env, "HELICOPTER_TENSOR_PARALLEL_SIZE"),
            infer.get("tensor_parallel_size"),
            gpu.get("tensor_parallel_size"),
        ),
        "--gpu-memory-utilization": pick(
            args.gpu_memory_utilization,
            infer.get("gpu_memory_utilization"),
        ),
        "--max-model-len": pick(
            args.max_model_len,
            model.get("max_model_len"),
            infer.get("max_model_len"),
        ),
        "--max-num-seqs": pick(
            args.max_num_seqs,
            infer.get("max_num_seqs"),
        ),
        "--max-num-batched-tokens": pick(
            args.max_num_batched_tokens,
            infer.get("max_num_batched_tokens"),
        ),
    }
    for option, value in option_values.items():
        if value is not None:
            command.extend([option, str(value)])

    auto_tool_choice = pick(
        args.enable_auto_tool_choice,
        env_value(env, "VLLM_ENABLE_AUTO_TOOL_CHOICE"),
        infer.get("enable_auto_tool_choice"),
        default=False,
    )
    if auto_tool_choice if isinstance(auto_tool_choice, bool) else str(auto_tool_choice).strip().lower() in {"1", "true", "yes", "on"}:
        command.append("--enable-auto-tool-choice")

    shown_env: dict[str, str] = {}
    apply_rwkv_env(shown_env, wkv_mode=wkv_mode, emb_device=emb_device)
    plan_env = strip_vllm_env(env)
    plan_env.update(shown_env)
    return CommandPlan(command=command, cwd=root, shown_env=shown_env, env=plan_env)


def build_takeoff_plan(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> CommandPlan:
    if args.algorithm != "grpo":
        raise SystemExit("only grpo takeoff is supported for RWKV right now")

    model_path, _ = resolve_model_path(config, args.model, root=root, env=env)
    data_root = dataset_root(config, args.dataset, root=root, env=env)
    datasets = table(config, "datasets")
    dataset_value = datasets.get(args.dataset, {})
    dataset = dataset_value if isinstance(dataset_value, dict) else {}

    paths = table(config, "paths")
    gpu = table(config, "gpu")
    takeoff_common = table(config, "takeoff")
    takeoff_algo_value = takeoff_common.get(args.algorithm, {})
    takeoff_algo = takeoff_algo_value if isinstance(takeoff_algo_value, dict) else {}
    takeoff = {**takeoff_common, **takeoff_algo}

    verl_path = resolve_path(
        str(pick(paths.get("verl_path"), env_value(env, "HELICOPTER_VERL_PATH", "VERL_PATH"), "src/train/verl-rwkv")),
        root=root,
        env=env,
    )
    rwkv_lm_path = resolve_path(
        str(pick(paths.get("rwkv_lm_path"), env_value(env, "RWKV_LM_PATH", "HELICOPTER_RWKV_LM_PATH"), "src/train/rwkv-lm")),
        root=root,
        env=env,
    )
    vllm_rwkv_path = resolve_path(
        str(pick(paths.get("vllm_rwkv_path"), env_value(env, "HELICOPTER_VLLM_RWKV_PATH", "VLLM_RWKV_PATH"), "src/infer/vllm-rwkv")),
        root=root,
        env=env,
    )

    has_train_files = "train_files" in dataset or env_value(env, "TRAIN_FILES") is not None
    has_val_files = "val_files" in dataset or env_value(env, "VAL_FILES") is not None
    dataset_uses_explicit_files = has_train_files and has_val_files
    if not args.dry_run:
        for path, message in (
            (model_path, "RWKV checkpoint not found"),
            (rwkv_lm_path, "rwkv-lm repository not found"),
            (vllm_rwkv_path, "vllm-rwkv repository not found"),
        ):
            exists = path.is_dir() if "repository" in message or "root" in message else path.is_file()
            if not exists:
                raise SystemExit(f"{message}: {path}")
        if not dataset_uses_explicit_files and not data_root.is_dir():
            raise SystemExit(f"dataset root not found: {data_root}")

    wkv_mode = str(
        pick(
            args.wkv_mode,
            env_value(env, "HELICOPTER_TAKEOFF_WKV_MODE", "VLLM_RWKV7_WKV_MODE"),
            takeoff.get("wkv_mode"),
            default="fp32io16",
        )
    )
    emb_device_value = pick(
        args.emb_device,
        env_value(env, "HELICOPTER_TAKEOFF_EMB_DEVICE"),
        takeoff.get("emb_device"),
        default="gpu",
    )
    emb_device = str(emb_device_value) if emb_device_value is not None else None
    num_nodes = pick(
        args.num_nodes,
        env_value(env, "HELICOPTER_NUM_NODES", "NNODES"),
        gpu.get("num_nodes"),
        takeoff.get("num_nodes"),
        default=1,
    )
    num_devices = pick(
        args.num_devices,
        env_value(env, "HELICOPTER_NUM_DEVICES", "NGPUS_PER_NODE"),
        gpu.get("num_devices"),
        takeoff.get("num_devices"),
        default=8,
    )

    python = python_executable(config, root=root, env=env, require_configured=True)
    shown_env: dict[str, str] = {}
    apply_rwkv_env(shown_env, wkv_mode=wkv_mode, emb_device=emb_device)
    shown_env["PYTHON"] = python
    shown_env["RWKV_MODEL_PATH"] = str(model_path)
    shown_env["RWKV_LM_PATH"] = str(rwkv_lm_path)
    plan_env = strip_vllm_env(env)
    plan_env.update(shown_env)
    current_pythonpath = plan_env.get("PYTHONPATH")
    plan_env["PYTHONPATH"] = (
        f"{vllm_rwkv_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(vllm_rwkv_path)
    )
    shown_env["PYTHONPATH"] = plan_env["PYTHONPATH"]

    command = [
        python,
        "-m",
        "verl.experimental.one_step_off_policy.main_ppo",
        *build_grpo_hydra_overrides(
            model_path=model_path,
            data_root=data_root,
            dataset=dataset,
            takeoff=takeoff,
            env=env,
            root=root,
            verl_path=verl_path,
            rwkv_lm_path=rwkv_lm_path,
            num_nodes=num_nodes,
            num_devices=num_devices,
        ),
        *(args.override or []),
    ]
    return CommandPlan(command=command, cwd=verl_path, shown_env=shown_env, env=plan_env)
