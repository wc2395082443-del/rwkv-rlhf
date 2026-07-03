# Native rwkv-lm Engine

This package is the training-side adapter for the upstream `rwkv-lm`
repository.

The intended implementation should preserve rwkv-lm's native behavior:
Lightning/DeepSpeed entrypoints, `.pth` checkpoints, CUDA extension environment
variables, precision flags, and optimizer scheduling. It should not turn RWKV
into a fake HuggingFace/FSDP model just to fit the existing transformer engine
path.

The package is registered with `EngineRegistry` as the `rwkv_lm` language-model
backend. Native import, args/env, checkpoint, weight, batch, and loss bridges
are available; model construction and optimizer creation delegate to upstream
rwkv-lm code.
