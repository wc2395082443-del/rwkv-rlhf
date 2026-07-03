import pandas as pd
for path in ['/root/autodl-tmp/data/verl_math500_smoke/train.parquet','/root/autodl-tmp/data/verl_math500_smoke/test.parquet']:
    df = pd.read_parquet(path)
    print('FILE', path)
    print('cols', list(df.columns))
    row = df.iloc[0]
    for k in ['data_source','prompt','ability','reward_model','extra_info']:
        if k in df.columns:
            print('KEY', k, 'TYPE', type(row[k]).__name__, 'VAL', row[k])
    print('N', len(df))
    print('---')
