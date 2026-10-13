import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path('/hy-tmp/evals/ultramath_step1000_pass16_20260910')
launcher_pid = 479397
evaluator_pid = 479443


def alive(pid):
    try:
        status = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return status != 'Z'
    except FileNotFoundError:
        return False


def main():
    command = Path(f'/proc/{launcher_pid}/cmdline').read_bytes()
    if b'/hy-tmp/run_ultramath_pass16.py' not in command:
        raise RuntimeError('Launcher identity mismatch')
    os.kill(launcher_pid, signal.SIGSTOP)
    print('WAITING_FOR_CHUNK_0000: only launcher paused; evaluator continues', flush=True)
    terminated = False
    try:
        while alive(evaluator_pid):
            time.sleep(10)
        summary = json.loads((root / 'chunk_0000/summary.json').read_text())
        if summary['num_tasks'] != 256 or summary['total_generations'] != 4096:
            raise RuntimeError('Incomplete first chunk; refusing handoff')
        os.kill(launcher_pid, signal.SIGKILL)
        terminated = True
        with open('/hy-tmp/ultramath_pass16_launcher.log', 'a') as output:
            process = subprocess.Popen(['/usr/local/bin/python3', '/hy-tmp/run_ultramath_pass16.py'],
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (root / 'optimization_handoff.json').write_text(json.dumps({
            'switched_at': time.time(), 'new_launcher_pid': process.pid,
            'batch_size': 1024, 'first_optimized_chunk': 1,
            'preserved_completed_chunk': 0, 'change': 'bulk token assignment; batch 512 to 1024'
        }, indent=2))
        print(f'SWITCHED launcher={process.pid} batch=1024', flush=True)
    finally:
        if not terminated and alive(launcher_pid):
            os.kill(launcher_pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
