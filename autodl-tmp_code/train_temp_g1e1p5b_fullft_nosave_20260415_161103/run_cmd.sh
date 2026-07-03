export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/RWKV-LM/RWKV-v7/train_temp
/root/miniconda3/bin/python3 train.py \
  --load_model /root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth \
  --proj_dir "/root/autodl-tmp/train_temp_g1e1p5b_fullft_nosave_20260415_161103" \
  --my_testing x070 \
  --ctx_len 128 \
  --train_stage 0 \
  --epoch_begin 0 \
  --data_file /root/RWKV-LM/RWKV-v5/demo \
  --my_exit_tokens 12800 \
  --magic_prime 50123 \
  --num_nodes 1 \
  --micro_bsz 1 \
  --n_layer 24 \
  --n_embd 2048 \
  --lr_init 1e-5 \
  --lr_final 1e-5 \
  --warmup_steps 1 \
  --beta1 0.9 \
  --beta2 0.99 \
  --adam_eps 1e-18 \
  --data_type binidx \
  --vocab_size 65536 \
  --weight_decay 0.001 \
  --epoch_save 1000 \
  --head_size 64 \
  --accelerator gpu \
  --devices 1 \
  --precision bf16 \
  --strategy deepspeed_stage_3_offload \
  --grad_cp 1 \
  --enable_progress_bar True \
  --ds_bucket_mb 2 \
  --no_save_checkpoint 1
