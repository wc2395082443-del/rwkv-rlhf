# Overview
This repository is build based on a from simple-eval


# OpenAI API Key
```
config key
```

# Eval
```
vllm serve TIGER-Lab/General-Reasoner-7B-preview --tensor-parallel-size 4
```

```
python -m evaluation.simple-evals.run_simple_evals_qwen --model General-Reasoner-7B-preview
```

```
python -m evaluation.eval_bbeh --model_path TIGER-Lab/General-Reasoner-7B-preview --output_file output-bbeh-General-Reasoner-7B-preview.json
```