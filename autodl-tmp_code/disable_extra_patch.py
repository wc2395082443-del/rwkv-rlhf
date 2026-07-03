from pathlib import Path
path = Path('/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py')
text = path.read_text(encoding='utf-8')
old1 = """        pending_extra = getattr(self, '_pending_extra_batch', None)\n        run_extra_only = bool(pending_extra and pending_extra.get('items'))\n        if run_extra_only:\n            self._pending_extra_batch = None\n"""
new1 = """        pending_extra = None\n        self._pending_extra_batch = None\n        run_extra_only = False\n"""
old2 = """            extra_target_samples = 0 if not hard_buffer_enabled else max(self.cfg.samples_per_question, int(self.cfg.hard_buffer_target_samples))\n            needed_questions = 0 if extra_target_samples <= 0 else max(1, int(math.ceil(float(extra_target_samples) / float(extra_group_size))))\n            if hard_buffer_enabled and needed_questions > 0:\n                hard_selected, hard_eligible = self._pop_hard_batch(step, needed_questions)\n            else:\n                hard_selected, hard_eligible = [], 0\n            hard_triggered = int(len(hard_selected) > 0)\n            if hard_triggered:\n                self._pending_extra_batch = {\n                    'queued_at_step': int(step),\n                    'group_size': int(extra_group_size),\n                    'items': [\n                        {\n                            'train_idx': int(item['train_idx']),\n                            'problem': item['problem'],\n                            'answer': item['answer'],\n                            'prompt_tokens': item['prompt_tokens'],\n                            'extra_source': 'hard',\n                        }\n                        for item in hard_selected\n                    ],\n                }\n"""
new2 = """            hard_selected, hard_eligible = [], 0\n            hard_triggered = 0\n            self._pending_extra_batch = None\n"""
if old1 not in text:
    raise SystemExit('patch block 1 not found')
if old2 not in text:
    raise SystemExit('patch block 2 not found')
text = text.replace(old1, new1, 1)
text = text.replace(old2, new2, 1)
path.write_text(text, encoding='utf-8')
print('patched', path)

