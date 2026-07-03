from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='Qwen/Qwen2.5-0.5B-Instruct',
    local_dir='/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct',
    local_dir_use_symlinks=False,
)
print(path)