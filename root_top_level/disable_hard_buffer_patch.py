from pathlib import Path
p = Path("/root/RWKV-LM/llama_grpo_baseline_v1/train.py")
text = p.read_text(encoding="utf-8")
old = """        hard_buffer_added = 0
        for group in group_infos:
            if group["correct_in_group"] == 0:
                added = self._push_hard_candidate(
                    train_idx=int(group["train_idx"]),
                    prompt_tokens=group["prompt_tokens"],
                    problem=group["problem"],
                    answer=group["answer"],
                    step=step,
                )
                if added:
                    hard_buffer_added += 1

        extra_target_samples = max(self.cfg.samples_per_question, int(self.cfg.hard_buffer_target_samples))
        extra_group_size = max(1, int(self.cfg.hard_buffer_group_size))
        needed_questions = max(1, int(math.ceil(float(extra_target_samples) / float(extra_group_size))))

        # Match the older hard-buffer behavior: only trigger when a full hard batch can be formed.
        hard_selected, hard_eligible = self._pop_hard_batch(step, needed_questions)
        hard_triggered = len(hard_selected) > 0
"""
new = """        hard_buffer_enabled = int(self.cfg.hard_buffer_target_samples) > 0
        hard_buffer_added = 0
        if hard_buffer_enabled:
            for group in group_infos:
                if group["correct_in_group"] == 0:
                    added = self._push_hard_candidate(
                        train_idx=int(group["train_idx"]),
                        prompt_tokens=group["prompt_tokens"],
                        problem=group["problem"],
                        answer=group["answer"],
                        step=step,
                    )
                    if added:
                        hard_buffer_added += 1

        if hard_buffer_enabled:
            extra_target_samples = max(self.cfg.samples_per_question, int(self.cfg.hard_buffer_target_samples))
            extra_group_size = max(1, int(self.cfg.hard_buffer_group_size))
            needed_questions = max(1, int(math.ceil(float(extra_target_samples) / float(extra_group_size))))

            # Match the older hard-buffer behavior: only trigger when a full hard batch can be formed.
            hard_selected, hard_eligible = self._pop_hard_batch(step, needed_questions)
            hard_triggered = len(hard_selected) > 0
        else:
            hard_selected = []
            hard_eligible = 0
            hard_triggered = False
"""
if old not in text:
    raise SystemExit("target block not found")
p.write_text(text.replace(old, new), encoding="utf-8")
