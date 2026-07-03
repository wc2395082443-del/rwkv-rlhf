# `root_verl/`

veRL framework snapshot used as a reference implementation for RLVR/GRPO-style training.

## Role In This Project

- Provided reference GRPO/PPO infrastructure and advantage-estimator logic.
- Used for comparing algorithm details such as group advantage normalization, KL terms, rollout layout, and trainer organization.
- Relevant to MaxRL investigation because related code paths expose alternative advantage estimators.

This is a framework snapshot, not the main RWKV experiment workspace. Most project-specific runs are under `autodl-tmp_code/`.
