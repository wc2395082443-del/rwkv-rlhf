from pathlib import Path
path = Path('/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py')
text = path.read_text(encoding='utf-8')
old = """        train_data: List[Dict[str, Any]],\n        test_data: List[Dict[str, Any]],\n        full_test_data: Optional[List[Dict[str, Any]]] = None,\n        out_dir: str,\n        device: str,\n        cfg: GRPOConfig,\n        seed: int = 42,\n"""
new = """        train_data: List[Dict[str, Any]],\n        test_data: List[Dict[str, Any]],\n        out_dir: str,\n        device: str,\n        cfg: GRPOConfig,\n        seed: int = 42,\n        full_test_data: Optional[List[Dict[str, Any]]] = None,\n"""
if old not in text:
    raise SystemExit('signature block not found')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
print('fixed signature order')

