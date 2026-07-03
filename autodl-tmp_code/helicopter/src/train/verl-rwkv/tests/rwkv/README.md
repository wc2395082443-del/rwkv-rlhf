# RWKV CPU Tests

These tests cover CPU-only RWKV integration contracts:

- native `rwkv-lm` actor/ref helper behavior
- RWKV tokenizer, reward, checkpoint, and weight mapping boundaries
- canonical `vllm` rollout registration for RWKV examples

They should remain CPU-only unless a test explicitly opts into a GPU or native
runtime dependency.
