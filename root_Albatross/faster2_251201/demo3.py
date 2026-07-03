import argparse
import codecs
import multiprocessing as mp
import os
import queue
import sys
import time
import types
import unicodedata
from collections import deque
from functools import lru_cache
from pathlib import Path


GRID_W, GRID_H = 20, 16
DEFAULT_MODEL_NAME = "/home/rwkv/rwkv7-g1f-7.2b-20260414-ctx8192"
TITLE_MODEL_NAME = "RWKV-7 7.2B"
TITLE_PRECISION = "FP16"
TITLE_GPU_NAME = "RTX 5090"
DEFAULT_SAMPLER_NOISE = 3.0
DEFAULT_GENERATION_LENGTH = 1000


def make_prompts(batch_size):
    return [
        ["Assistant: <think" for _ in range(batch_size)],
        ["Assistant: <think>嗯" for _ in range(batch_size)],
        ["Assistant: <think>私" for _ in range(batch_size)],
    ]


@lru_cache(maxsize=4096)
def char_cells(ch):
    if not ch:
        return 0
    code = ord(ch)
    if ch == "\t":
        return 1
    if code < 32 or 0x7F <= code < 0xA0:
        return 0
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Cf", "Mn", "Me"):
        return 0
    if unicodedata.east_asian_width(ch) in ("F", "W"):
        return 2
    return 1


def put_drop(out_q, msg):
    try:
        out_q.put_nowait(msg)
        return False
    except queue.Full:
        return True


def make_model_args(model_name):
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = model_name
    return args


def flush_pending_text(out_q, pending, force, flush_dt, last_flush, dropped_frames):
    pending_chars = sum(sum(len(chunk) for chunk in chunks) for chunks in pending)
    if pending_chars <= 0:
        return last_flush, dropped_frames
    now = time.perf_counter()
    if not force and now - last_flush < flush_dt:
        return last_flush, dropped_frames

    updates = []
    for panel_idx, chunks in enumerate(pending):
        if chunks:
            updates.append((panel_idx, "".join(chunks)))
            chunks.clear()
    if updates and put_drop(out_q, ("batch_text", updates)):
        dropped_frames += 1
    return now, dropped_frames


def model_process(out_q, ctrl_q, shutdown_event, cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from reference.rwkv7 import RWKV_x070
    from reference.utils import TRIE_TOKENIZER, sampler_simple_batch

    grid_w = cfg["grid_w"]
    grid_h = cfg["grid_h"]
    batch_size = grid_w * grid_h
    prompts = make_prompts(batch_size)
    current_prompt = 0
    model_name = cfg["model_name"]
    generation_length = cfg["generation_length"]
    sampler_noise = cfg["sampler_noise"]
    flush_hz = max(1, cfg["ui_flush_hz"])
    flush_dt = 1.0 / flush_hz
    perf_interval = max(1, cfg["perf_interval"])
    perf_sync_interval = max(0, cfg["perf_sync_interval"])

    if cfg.get("model_nice", 0) > 0:
        try:
            os.nice(int(cfg["model_nice"]))
        except OSError:
            pass

    state = None
    dropped_frames = 0

    try:
        model = RWKV_x070(make_model_args(model_name))
        tokenizer = TRIE_TOKENIZER(str(Path(__file__).resolve().parent / "reference" / "rwkv_vocab_v20230424.txt"))

        while not shutdown_event.is_set():
            try:
                if state is not None:
                    del state
                    torch.cuda.empty_cache()

                state = model.generate_zero_state(batch_size)
                put_drop(out_q, ("clear_all", None))
                put_drop(out_q, ("batch_text", list(enumerate(prompts[current_prompt]))))

                out = model.forward_batch([tokenizer.encode(x) for x in prompts[current_prompt]], state)
                decoders = [codecs.getincrementaldecoder("utf-8")("strict") for _ in range(batch_size)]
                pending = [[] for _ in range(batch_size)]
                last_flush = time.perf_counter()
                perf_start = last_flush
                perf_tokens = 0

                for step in range(generation_length):
                    if shutdown_event.is_set():
                        break

                    try:
                        while True:
                            msg = ctrl_q.get_nowait()
                            if msg == "switch_prompt":
                                last_flush, dropped_frames = flush_pending_text(
                                    out_q, pending, True, flush_dt, last_flush, dropped_frames
                                )
                                current_prompt = (current_prompt + 1) % len(prompts)
                                put_drop(out_q, ("perf", f"Switched to prompt {current_prompt + 1}: {prompts[current_prompt][0][:50]}..."))
                                raise StopIteration
                    except queue.Empty:
                        pass

                    new_tokens_tensor = sampler_simple_batch(out, sampler_noise)
                    flat_tokens = new_tokens_tensor.view(-1).detach().cpu().tolist()

                    # Keep the next model input on GPU; only copy token ids for display decode.
                    if hasattr(model, "forward_seq_batch_1"):
                        out = model.forward_seq_batch_1(new_tokens_tensor, state, False)
                    else:
                        out = model.forward_batch([[int(x)] for x in flat_tokens], state)
                    perf_tokens += batch_size

                    for idx, token_id in enumerate(flat_tokens):
                        text = decoders[idx].decode(tokenizer.idx2token[token_id], final=False)
                        if text:
                            pending[idx].append(text)

                    last_flush, dropped_frames = flush_pending_text(
                        out_q, pending, False, flush_dt, last_flush, dropped_frames
                    )

                    if (step + 1) % perf_interval == 0:
                        if perf_sync_interval and (step + 1) % perf_sync_interval == 0:
                            torch.cuda.synchronize()
                        now = time.perf_counter()
                        elapsed = max(1e-9, now - perf_start)
                        tps = round(perf_tokens / elapsed)
                        perf_tokens = 0
                        perf_start = now
                        put_drop(
                            out_q,
                            (
                                "perf",
                                f"{TITLE_MODEL_NAME} {TITLE_PRECISION} bsz{batch_size} @ {TITLE_GPU_NAME} | "
                                f"Token/s {tps} | UI flush {flush_hz}Hz | dropped UI frames {dropped_frames} | "
                                'Press "a" to switch prompt',
                            ),
                        )

                last_flush, dropped_frames = flush_pending_text(
                    out_q, pending, True, flush_dt, last_flush, dropped_frames
                )
            except StopIteration:
                continue
    except Exception as exc:
        put_drop(out_q, ("error", f"{type(exc).__name__}: {exc}"))
    finally:
        put_drop(out_q, ("done", None))


class PanelBuffer:
    __slots__ = ("cols", "rows", "lines", "cur", "cur_cells", "dirty")

    def __init__(self, cols, rows, history=128):
        self.cols = max(1, cols)
        self.rows = max(1, rows)
        self.lines = deque(maxlen=history)
        self.cur = []
        self.cur_cells = 0
        self.dirty = True

    def resize(self, cols, rows):
        self.cols = max(1, cols)
        self.rows = max(1, rows)
        self.dirty = True

    def clear(self):
        self.lines.clear()
        self.cur = []
        self.cur_cells = 0
        self.dirty = True

    def append(self, text):
        cols = self.cols
        for ch in text:
            if ch == "\n":
                self.lines.append("".join(self.cur))
                self.cur = []
                self.cur_cells = 0
                continue
            width = char_cells(ch)
            if self.cur and self.cur_cells + width > cols:
                self.lines.append("".join(self.cur))
                self.cur = []
                self.cur_cells = 0
            self.cur.append(ch)
            self.cur_cells += width
            if self.cur_cells >= cols:
                self.lines.append("".join(self.cur))
                self.cur = []
                self.cur_cells = 0
        self.dirty = True

    def visible(self):
        take = self.rows - (1 if self.cur else 0)
        if take > 0:
            newest = []
            for line in reversed(self.lines):
                newest.append(line)
                if len(newest) >= take:
                    break
            result = list(reversed(newest))
        else:
            result = []
        if self.cur:
            result.append("".join(self.cur))
        if len(result) < self.rows:
            result = [""] * (self.rows - len(result)) + result
        return result[-self.rows :]


def import_qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        return QtCore, QtGui, QtWidgets
    except ModuleNotFoundError:
        from PySide2 import QtCore, QtGui, QtWidgets

        return QtCore, QtGui, QtWidgets


def gui_process(in_q, ctrl_q, shutdown_event, cfg):
    if cfg.get("gui_nice", 0) > 0:
        try:
            os.nice(int(cfg["gui_nice"]))
        except OSError:
            pass

    try:
        QtCore, QtGui, QtWidgets = import_qt()
    except Exception as exc:
        print(f"Qt import failed: {exc}", file=sys.stderr, flush=True)
        shutdown_event.set()
        return

    class GridWidget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.grid_w = cfg["grid_w"]
            self.grid_h = cfg["grid_h"]
            self.num_panels = self.grid_w * self.grid_h
            self.status = "Loading model ..."
            self.font = QtGui.QFont(cfg["font_family"], cfg["font_size"])
            self.font.setStyleHint(QtGui.QFont.Monospace)
            self.font.setFixedPitch(True)
            self.metrics = QtGui.QFontMetrics(self.font)
            self.line_h = max(1, self.metrics.lineSpacing())
            self.char_w = max(1, self.metrics.horizontalAdvance("M") if hasattr(self.metrics, "horizontalAdvance") else self.metrics.width("M"))
            self.status_h = self.line_h + 8
            self.cell_rects = []
            self.colors = [
                QtGui.QColor(235, 235, 235),
                QtGui.QColor(110, 220, 235),
                QtGui.QColor(120, 230, 150),
            ]
            self.bg = QtGui.QColor(8, 10, 12)
            self.status_color = QtGui.QColor(255, 95, 95)
            self.panels = []
            self.dirty_panels = set(range(self.num_panels))
            self.status_dirty = True
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.setWindowTitle(f"{TITLE_MODEL_NAME} {TITLE_PRECISION} demo3")
            self.resize(cfg["window_w"], cfg["window_h"])
            self.rebuild_layout()

            self.poll_timer = QtCore.QTimer(self)
            self.poll_timer.timeout.connect(self.poll_messages)
            self.poll_timer.start(max(1, cfg["poll_ms"]))

            self.paint_timer = QtCore.QTimer(self)
            self.paint_timer.timeout.connect(self.flush_repaints)
            self.paint_timer.start(max(1, int(1000 / max(1, cfg["fps"]))))

        def rebuild_layout(self):
            width = max(1, self.width())
            height = max(1, self.height() - self.status_h)
            base_w = max(1, width // self.grid_w)
            base_h = max(1, height // self.grid_h)
            widths = [base_w + (1 if c < width - base_w * self.grid_w else 0) for c in range(self.grid_w)]
            heights = [base_h + (1 if r < height - base_h * self.grid_h else 0) for r in range(self.grid_h)]
            lefts = []
            acc = 0
            for w in widths:
                lefts.append(acc)
                acc += w
            tops = []
            acc = self.status_h
            for h in heights:
                tops.append(acc)
                acc += h
            self.cell_rects = []
            new_panels = []
            for r in range(self.grid_h):
                for c in range(self.grid_w):
                    rect = QtCore.QRect(lefts[c], tops[r], widths[c], heights[r])
                    self.cell_rects.append(rect)
                    cols = max(1, rect.width() // self.char_w)
                    rows = max(1, rect.height() // self.line_h)
                    idx = r * self.grid_w + c
                    if idx < len(self.panels):
                        panel = self.panels[idx]
                        panel.resize(cols, rows)
                    else:
                        panel = PanelBuffer(cols, rows)
                    new_panels.append(panel)
            self.panels = new_panels
            self.dirty_panels = set(range(self.num_panels))
            self.status_dirty = True
            self.update()

        def resizeEvent(self, event):
            self.rebuild_layout()
            super().resizeEvent(event)

        def keyPressEvent(self, event):
            key = event.key()
            if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
                shutdown_event.set()
                self.window().close()
            elif key == QtCore.Qt.Key_A:
                try:
                    ctrl_q.put_nowait("switch_prompt")
                except queue.Full:
                    pass
            else:
                super().keyPressEvent(event)

        def poll_messages(self):
            processed = 0
            max_messages = cfg["max_gui_messages"]
            while processed < max_messages:
                try:
                    msg_type, payload = in_q.get_nowait()
                except queue.Empty:
                    break
                processed += 1
                if msg_type == "batch_text":
                    for idx, text in payload:
                        if 0 <= idx < self.num_panels:
                            self.panels[idx].append(text)
                            self.dirty_panels.add(idx)
                elif msg_type == "clear_all":
                    for idx, panel in enumerate(self.panels):
                        panel.clear()
                        self.dirty_panels.add(idx)
                elif msg_type == "perf":
                    self.status = payload
                    self.status_dirty = True
                elif msg_type == "error":
                    self.status = f"Error: {payload}"
                    self.status_dirty = True
                elif msg_type == "done":
                    self.status = "Finished"
                    self.status_dirty = True
                    shutdown_event.set()
            if shutdown_event.is_set() and not self.dirty_panels:
                self.window().close()

        def flush_repaints(self):
            if self.status_dirty:
                self.update(QtCore.QRect(0, 0, self.width(), self.status_h))
                self.status_dirty = False
            if not self.dirty_panels:
                return
            region = QtGui.QRegion()
            for idx in self.dirty_panels:
                if 0 <= idx < len(self.cell_rects):
                    region = region.united(QtGui.QRegion(self.cell_rects[idx]))
                    self.panels[idx].dirty = False
            self.dirty_panels.clear()
            self.update(region)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setFont(self.font)
            painter.fillRect(event.rect(), self.bg)

            region = event.region()
            status_rect = QtCore.QRect(0, 0, self.width(), self.status_h)
            if region.intersects(status_rect):
                painter.setPen(self.status_color)
                painter.drawText(6, self.line_h + 2, self.status)

            for idx, rect in enumerate(self.cell_rects):
                if not region.intersects(rect):
                    continue
                row = idx // self.grid_w
                col = idx % self.grid_w
                painter.setPen(self.colors[(row + col) % len(self.colors)])
                y = rect.top() + self.metrics.ascent()
                for line in self.panels[idx].visible():
                    painter.drawText(rect.left(), y, line)
                    y += self.line_h
                    if y > rect.bottom() + self.line_h:
                        break
            painter.end()

    app = QtWidgets.QApplication(sys.argv[:1])
    widget = GridWidget()
    screen = app.primaryScreen()
    if screen is not None:
        widget.setGeometry(screen.availableGeometry())
    widget.setWindowState(widget.windowState() | QtCore.Qt.WindowMaximized)
    widget.show()
    QtCore.QTimer.singleShot(0, widget.showMaximized)
    QtCore.QTimer.singleShot(100, widget.showMaximized)
    rc = app.exec_() if hasattr(app, "exec_") else app.exec()
    shutdown_event.set()
    return rc


def parse_args():
    parser = argparse.ArgumentParser(description="Albatross demo3 Qt GUI renderer")
    parser.add_argument("--grid-w", type=int, default=GRID_W)
    parser.add_argument("--grid-h", type=int, default=GRID_H)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--generation-length", type=int, default=DEFAULT_GENERATION_LENGTH)
    parser.add_argument("--sampler-noise", type=float, default=DEFAULT_SAMPLER_NOISE)
    parser.add_argument("--fps", type=int, default=30, help="GUI repaint FPS")
    parser.add_argument("--ui-flush-hz", type=int, default=30, help="producer-to-GUI batch flush rate")
    parser.add_argument("--perf-interval", type=int, default=30)
    parser.add_argument("--perf-sync-interval", type=int, default=30, help="CUDA sync every N generation steps for reporting; 0 disables")
    parser.add_argument("--queue-size", type=int, default=4, help="small queue prevents GUI from backpressuring the model")
    parser.add_argument("--gui-nice", type=int, default=5, help="increase GUI process niceness so it yields CPU to the model")
    parser.add_argument("--model-nice", type=int, default=0, help="increase model process niceness; normally keep this at 0")
    parser.add_argument("--font-family", default="monospace")
    parser.add_argument("--font-size", type=int, default=9)
    parser.add_argument("--window-w", type=int, default=1600, help="initial size before maximize")
    parser.add_argument("--window-h", type=int, default=1000, help="initial size before maximize")
    parser.add_argument("--poll-ms", type=int, default=2)
    parser.add_argument("--max-gui-messages", type=int, default=256)
    return parser.parse_args()


def main():
    cli = parse_args()
    cfg = vars(cli)
    mp.set_start_method("spawn", force=True)

    text_q = mp.Queue(maxsize=max(1, cli.queue_size))
    ctrl_q = mp.Queue(maxsize=16)
    shutdown_event = mp.Event()

    producer = mp.Process(target=model_process, args=(text_q, ctrl_q, shutdown_event, cfg), daemon=False)
    gui = mp.Process(target=gui_process, args=(text_q, ctrl_q, shutdown_event, cfg), daemon=False)

    producer.start()
    gui.start()

    try:
        while producer.is_alive() and gui.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        shutdown_event.set()
        for proc in (producer, gui):
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join()


if __name__ == "__main__":
    main()
