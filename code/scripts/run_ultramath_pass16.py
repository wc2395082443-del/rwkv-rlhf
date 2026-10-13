import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/hy-tmp/evals/ultramath_step1000_pass16_20260910')
MODEL = Path('/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810/checkpoint_step_1000.pth')
SOURCE = '/hy-tmp/eval_ultramath_optimized.py'
ALBATROSS = '/root/eval_transfer/albatross/faster3a_2605'
ENDPOINT = 'https://hf-mirror.com'
REPO = 'openbmb/UltraData-RL-2609'


def fetch(url):
    for attempt in range(5):
        try:
            return subprocess.check_output(['curl', '-fL', '--connect-timeout', '15', '--max-time', '90', '-sS', url])
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    manifest_path = ROOT / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        info = json.loads(fetch(f'{ENDPOINT}/api/datasets/{REPO}'))
        manifest = {'dataset': REPO, 'revision': info['sha'], 'model': str(MODEL),
                    'started_at': time.time(), 'rollout': 16, 'max_new_tokens': 8192,
                    'temperature': 0.8, 'top_p': 0.35, 'top_k': 40,
                    'prompt_style': 'official_flower', 'stage_count': 1}
        save_json(manifest_path, manifest)
    evaluator = ROOT / 'eval_math500_flower_optimized.py'
    if not evaluator.exists():
        evaluator.write_bytes(Path(SOURCE).read_bytes())
    records = []
    hashes = {}
    for part in range(1, 5):
        name = f'Math_part-{part}-of-4.jsonl'
        target = ROOT / name
        if not target.exists():
            payload = fetch(f'{ENDPOINT}/datasets/{REPO}/resolve/{manifest["revision"]}/data/Math/{name}')
            temporary = target.with_suffix('.download')
            temporary.write_bytes(payload)
            temporary.replace(target)
        payload = target.read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
        print(f'DOWNLOADED {name} rows={len(rows)}', flush=True)
        records.extend(rows)
    identifiers = [row['uuid'] for row in records]
    if len(set(identifiers)) != len(records):
        raise ValueError('Duplicate UUIDs in source')
    for row in records:
        if not isinstance(row['query'], str) or not row['query'].strip():
            raise ValueError('Invalid question')
        if not isinstance(row['ground_truth'], str) or not row['ground_truth'].strip():
            raise ValueError('Invalid ground truth')
    manifest.update(num_tasks=len(records), total_generations=len(records) * 16, source_sha256=hashes)
    save_json(manifest_path, manifest)
    print(f'DATA_READY tasks={len(records)} generations={len(records)*16}', flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
                       TORCH_EXTENSIONS_DIR='/hy-tmp/torch_extensions_official_ckpt_eval_20260827',
                       PYTHONPATH=ALBATROSS + ':' + environment.get('PYTHONPATH', ''),
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    totals = {'num_tasks': 0, 'total_generations': 0, 'correct_generations': 0, 'passed_tasks': 0}
    for offset in range(0, len(records), 256):
        chunk = ROOT / f'chunk_{offset//256:04d}'
        chunk.mkdir(exist_ok=True)
        dataset = chunk / 'dataset.jsonl'
        expected = records[offset:offset+256]
        if not dataset.exists():
            with dataset.open('w') as handle:
                for row in expected:
                    handle.write(json.dumps({'problem': row['query'], 'answer': row['ground_truth'],
                                             'unique_id': row['uuid'], 'subject': row.get('source', '')}, ensure_ascii=False) + '\n')
        summary_path = chunk / 'summary.json'
        if not summary_path.exists():
            print(f'CHUNK_START offset={offset} count={len(expected)}', flush=True)
            command = [sys.executable, str(evaluator), '--albatross-dir', ALBATROSS,
                       '--model', str(MODEL), '--dataset', str(dataset), '--out-dir', str(chunk),
                       '--gpus', '0', '--rollout', '16', '--bsz', '1024', '--max-new-tokens', '8192',
                       '--ctx-limit', '12288', '--temperature', '0.8', '--top-p', '0.35', '--top-k', '40',
                       '--seed', str(42 + offset//256), '--prompt-style', 'official_flower',
                       '--torch-extensions-dir', environment['TORCH_EXTENSIONS_DIR'],
                       '--progress-every', '100', '--verify-workers', '8']
            save_json(chunk / 'command.json', command)
            with (chunk / 'eval.log').open('a') as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads(summary_path.read_text())
        if summary['num_tasks'] != len(expected) or summary['total_generations'] != len(expected)*16:
            raise ValueError(f'Incomplete chunk {chunk}')
        for key in ('num_tasks', 'total_generations', 'correct_generations'):
            totals[key] += summary[key]
        totals['passed_tasks'] += round(summary['pass_at_rollout_accuracy'] * summary['num_tasks'])
        progress = dict(totals, avg_at_16=totals['correct_generations']/totals['total_generations'],
                        pass_at_16=totals['passed_tasks']/totals['num_tasks'],
                        target_tasks=len(records), elapsed_sec=time.time()-manifest['started_at'],
                        updated_at=time.time(), complete=totals['num_tasks']==len(records))
        save_json(ROOT / 'progress.json', progress)
        print('PROGRESS ' + json.dumps(progress), flush=True)
    save_json(ROOT / 'summary.json', progress)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
