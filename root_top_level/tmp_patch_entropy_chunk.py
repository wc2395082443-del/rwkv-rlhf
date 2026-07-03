from pathlib import Path
path = Path('/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py')
text = path.read_text(encoding='utf-8')
old = """                    top_k = min(500, logits.size(-1))
                    top_logits, _ = torch.topk(logits.float(), k=top_k, dim=-1)
                    logp_top = top_logits.float() - logsumexp.unsqueeze(-1)
                    p_top = torch.exp(logp_top)
                    entropy_per_token = -(p_top * logp_top).sum(dim=-1)
                    del top_logits, logp_top, p_top, logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
"""
new = """                    # Avoid materializing a full fp32 vocab tensor here. This block is only
                    # for logging entropy, so we stream top-k selection over vocab chunks.
                    top_k = min(500, logits.size(-1))
                    top_logits = None
                    vocab_chunk = 2048
                    for v_start in range(0, logits.size(-1), vocab_chunk):
                        chunk = logits[..., v_start:v_start + vocab_chunk].float()
                        k_chunk = min(top_k, chunk.size(-1))
                        chunk_top, _ = torch.topk(chunk, k=k_chunk, dim=-1)
                        if top_logits is None:
                            top_logits = chunk_top
                        else:
                            merged_top = torch.cat((top_logits, chunk_top), dim=-1)
                            top_logits, _ = torch.topk(merged_top, k=top_k, dim=-1)
                            del merged_top
                        del chunk, chunk_top
                    logp_top = top_logits - logsumexp.unsqueeze(-1)
                    p_top = torch.exp(logp_top)
                    entropy_per_token = -(p_top * logp_top).sum(dim=-1)
                    del top_logits, logp_top, p_top, logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
"""
if old not in text:
    raise SystemExit('target block not found')
backup = path.with_name(path.name + '.bak_entropy_chunk_20260503')
backup.write_text(text, encoding='utf-8')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('patched', path)
print('backup', backup)

