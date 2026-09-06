# RWKV RLHF Experiments

This branch contains a cleaned project snapshot for **DAPO-gsm8k** on RWKV7 g1i 1.5B.

See `DAPO-gsm8k/README.md` for code, commands, and results.

## Latest Training Curve

The extended DAPO run continues from the GSM8K-trained `step155` checkpoint to global `step1300` on processed `MATH17K` with the official RWKV flower prompt.

Updated: 2026-09-06. This section describes the continuation run and is separate from the earlier GSM8K `step155` report.

![DAPO global training curves](https://raw.githubusercontent.com/wc2395082443-del/rwkv-rlhf/DAPO-gsm8k/DAPO-gsm8k/results/g1i_dapo_global_1_1300_sixpanel.jpg)

![One-stage benchmark curves](https://raw.githubusercontent.com/wc2395082443-del/rwkv-rlhf/DAPO-gsm8k/results/benchmark_curves_one_stage_step0_compact.jpg)
