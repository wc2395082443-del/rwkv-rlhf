import importlib.util
from pathlib import Path


PATCH_MODULE = Path(__file__).resolve().parents[2] / "verl" / "utils" / "vllm" / "patch.py"
spec = importlib.util.spec_from_file_location("verl_vllm_patch", PATCH_MODULE)
patch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(patch)


def test_moe_weight_loader_patch_ignores_direct_non_moe_model(monkeypatch):
    class SupportedMoeModel:
        pass

    class DirectNonMoeModel:
        pass

    monkeypatch.setattr(patch, "SUPPORTED_MOE_MODELS", [SupportedMoeModel])

    patch.patch_vllm_moe_model_weight_loader(DirectNonMoeModel())
