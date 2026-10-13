import html
import json
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

base = Path('/hy-tmp/evals/missing_five_benchmarks_checkpoints_20260905')
baseline_base = Path('/hy-tmp/evals/g1i_baseline_unified_20260906')
out = base / 'progress.html'
tsv = base / 'progress.tsv'
tasks = []
gpu_history = deque()
throughput_history = deque()
last_progress = None
for step in [1050, 1100, 1150, 1250]:
    tasks.append((step, 'gsm8k'))
for step in [1050, 1100, 1150, 1200, 1250]:
    tasks.append((step, 'math500'))
for step in [85, 100, 105, 155, 200, 300, 1050, 1100, 1150, 1200, 1250, 1300]:
    for bench in ['aime24', 'aime25', 'amc23']:
        tasks.append((step, bench))

def read_status():
    result = {}
    starts = {}
    durations = []
    path = base / 'status.tsv'
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] in {'START', 'DONE'}:
                key = (int(parts[3]), parts[4])
                stamp = datetime.strptime(f'{parts[1]} {parts[2]}', '%Y-%m-%d %H:%M:%S')
                if parts[0] == 'START':
                    starts[key] = stamp
                    result[key] = 'running'
                elif key in starts:
                    result[key] = 'done'
                    durations.append((stamp - starts[key]).total_seconds())
    return result, starts, durations

def read_gpu():
    try:
        raw = subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], text=True, timeout=5).strip().splitlines()[0]
        gpu, bandwidth, used, total = [float(x.strip()) for x in raw.split(',')]
        return gpu, used / total * 100, bandwidth, used, total
    except Exception:
        return None

def gpu_average(seconds):
    cutoff = time.time() - seconds
    values = [item for item in gpu_history if item[0] >= cutoff]
    if not values:
        return None
    return tuple(sum(item[index] for item in values) / len(values) for index in range(1, 4))

def read_throughput():
    logs = sorted(base.glob('step*/**/eval.log'), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    text = logs[-1].read_text(errors='replace')
    tps = re.findall(r'\btps=([0-9.]+)', text)
    progress = re.findall(r'done=(\d+)/(\d+)', text)
    if not tps or not progress:
        return None
    return str(logs[-1]), float(tps[-1]), int(progress[-1][0])

def throughput_average(seconds):
    cutoff = time.time() - seconds
    values = [item for item in throughput_history if item[0] >= cutoff]
    if not values:
        return None
    token_avg = sum(item[1] * item[4] for item in values) / sum(item[4] for item in values)
    sample_total = sum(item[3] for item in values)
    sample_time = sum(item[4] for item in values)
    sample_avg = sample_total / sample_time if sample_time else None
    return token_avg, sample_avg

def task_progress(step, bench, statuses):
    task = base / f'step{step}' / bench
    summary = task / 'summary.json'
    if summary.exists() and summary.stat().st_size:
        data = json.loads(summary.read_text())
        correct = data.get('correct_generations')
        total = data.get('total_generations')
        count = f'{correct}/{total} ({correct / total * 100:.2f}%)' if correct is not None and total else ''
        return 'done', 100, count
    if statuses.get((step, bench)) != 'running':
        return 'pending', 0, ''
    log = task / 'eval.log'
    if log.exists():
        matches = re.findall(r'done=(\d+)/(\d+)', log.read_text(errors='replace'))
        if matches:
            done, total = map(int, matches[-1])
            correct = re.findall(r'(?:correct|correct_generations)=(\d+)', log.read_text(errors='replace'))
            count = f'{correct[-1]}/{done} ({int(correct[-1]) / done * 100:.2f}%)' if correct and done else f'-/{done}'
            return 'running', round(done * 100 / total, 1) if total else 0, count
    return 'running', 0, ''

def baseline_progress(bench):
    task = baseline_base / bench
    summary = task / 'summary.json'
    if summary.exists() and summary.stat().st_size:
        data = json.loads(summary.read_text())
        correct = data.get('correct_generations')
        total = data.get('total_generations')
        detail = f'{correct}/{total} ({correct / total * 100:.2f}%)' if correct is not None and total else ''
        return 'done', 100, detail
    log = task / 'eval.log'
    if log.exists():
        matches = re.findall(r'done=(\d+)/(\d+)', log.read_text(errors='replace'))
        if matches:
            done, total = map(int, matches[-1])
            return 'running', round(done * 100 / total, 1) if total else 0, f'-/{done}'
    return 'pending', 0, ''

while True:
    current_gpu = read_gpu()
    if current_gpu:
        gpu_history.append((time.time(), *current_gpu[:3]))
    while gpu_history and gpu_history[0][0] < time.time() - 900:
        gpu_history.popleft()
    progress = read_throughput()
    if progress:
        now_ts = time.time()
        sample_rate = None
        sample_delta = 0
        delta_t = 0
        log_path, token_tps, done = progress
        if last_progress and last_progress[0] == log_path:
            _, previous_ts, previous_done = last_progress
            delta_t = now_ts - previous_ts
            if delta_t > 0 and done >= previous_done:
                sample_delta = done - previous_done
                sample_rate = sample_delta / delta_t
        last_progress = (log_path, now_ts, done)
        throughput_history.append((now_ts, token_tps, sample_rate, sample_delta, delta_t or 10))
    while throughput_history and throughput_history[0][0] < time.time() - 900:
        throughput_history.popleft()
    statuses, starts, durations = read_status()
    rows = []
    counts = {'done': 0, 'running': 0, 'pending': 0}
    for step, bench in tasks:
        status, percent, detail = task_progress(step, bench, statuses)
        counts[status] += 1
        rows.append((step, bench, status, percent, detail))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    active_starts = list(starts.values())
    first_start = min(active_starts) if active_starts else datetime.now()
    elapsed = max(0, int((datetime.now() - first_start).total_seconds()))
    done_count = counts['done']
    pending_count = counts['pending']
    running_count = counts['running']
    avg_task_sec = sum(durations) / len(durations) if durations else 0
    running_remaining = 0
    for step, bench, status, percent, _ in rows:
        if status == 'running' and percent > 0 and starts.get((step, bench)):
            task_elapsed = (datetime.now() - starts[(step, bench)]).total_seconds()
            running_remaining += max(0, task_elapsed * (100 - percent) / percent)
    remaining = running_remaining + pending_count * avg_task_sec
    def fmt_time(seconds):
        seconds = int(max(0, seconds))
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f'{hours}h {minutes}m {seconds}s'
    gpu_lines = []
    for label, seconds in [('1分钟', 60), ('5分钟', 300), ('15分钟', 900)]:
        average = gpu_average(seconds)
        if average:
            gpu, memory, bandwidth = average
            gpu_lines.append(f'{label}：GPU {gpu:.1f}% ｜ 显存 {memory:.1f}% ｜ 带宽 {bandwidth:.1f}%')
        else:
            gpu_lines.append(f'{label}：暂无数据')
    gpu_html = '<br>'.join(gpu_lines)
    throughput_lines = []
    for label, seconds in [('1分钟', 60), ('5分钟', 300), ('15分钟', 900)]:
        average = throughput_average(seconds)
        if average:
            sample_text = f'{average[1]:.2f}' if average[1] is not None else '计算中'
            throughput_lines.append(f'{label}：{average[0]:.1f} token/s ｜ {sample_text} sample/s')
        else:
            throughput_lines.append(f'{label}：暂无数据')
    throughput_html = '<br>'.join(throughput_lines)
    lines = ['step\tbenchmark\tstatus\tpercent\tdetail']
    lines += [f'{s}\t{b}\t{st}\t{p}\t{d}' for s, b, st, p, d in rows]
    tsv.write_text('\n'.join(lines) + '\n')
    trs = ''.join(
        f'<tr class="{st}"><td>{s}</td><td>{html.escape(b)}</td><td>{st}</td><td>{p:.1f}%</td><td>{d}</td></tr>'
        for s, b, st, p, d in rows
    )
    baseline_rows = []
    baseline_counts = {'done': 0, 'running': 0, 'pending': 0}
    for bench in ['gsm8k', 'math500', 'aime24', 'aime25', 'amc23']:
        status, percent, detail = baseline_progress(bench)
        baseline_counts[status] += 1
        baseline_rows.append(f'<tr class="{status}"><td>g1i</td><td>{bench}</td><td>{status}</td><td>{percent:.1f}%</td><td>{detail}</td></tr>')
    baseline_html = ''.join(baseline_rows)
    page = f'''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>Missing benchmark evaluation progress</title>
<style>body{{font:14px sans-serif;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 12px}}.done{{background:#e7f7e7}}.running{{background:#fff4cf}}.pending{{color:#777}}h1{{margin-bottom:6px}}</style>
<h1>Missing benchmark evaluation</h1><p><b>已耗时：{fmt_time(elapsed).upper()} &nbsp;&nbsp; 预估剩余时间：{fmt_time(remaining).upper()}</b></p><p><b>GPU利用率 / 显存占用率 / 带宽利用率平均值</b><br>{gpu_html}</p><p><b>吞吐平均值（token/s / sample/s）</b><br>{throughput_html}</p><p>Updated: {now} &nbsp; Done: {counts['done']} &nbsp; Running: {counts['running']} &nbsp; Pending: {counts['pending']} &nbsp; Total: {len(tasks)}</p>
<table><tr><th>Step</th><th>Benchmark</th><th>Status</th><th>Progress</th><th>Correct / Total samples (accuracy)</th></tr>{trs}</table>
<h2>g1i baseline</h2><p>Done: {baseline_counts['done']} &nbsp; Running: {baseline_counts['running']} &nbsp; Pending: {baseline_counts['pending']} &nbsp; Total: 5</p>
<table><tr><th>Model</th><th>Benchmark</th><th>Status</th><th>Progress</th><th>Correct / Total samples (accuracy)</th></tr>{baseline_html}</table>'''
    out.write_text(page)
    time.sleep(10)
