import pandas as pd
paths = [
    '/root/autodl-tmp/data/verl_math500/train.parquet',
    '/root/autodl-tmp/data/verl_math500/test.parquet',
    '/root/autodl-tmp/data/verl_math500_smoke/train.parquet',
    '/root/autodl-tmp/data/verl_math500_smoke/test.parquet',
]
for path in paths:
    df = pd.read_parquet(path)
    before = df['data_source'].value_counts(dropna=False).to_dict()
    df['data_source'] = 'HuggingFaceH4/MATH-500'
    df.to_parquet(path, index=False)
    after = df['data_source'].value_counts(dropna=False).to_dict()
    print(path)
    print('before', before)
    print('after', after)
