import argparse
import codecs
import html
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent)
DEFAULT_MODEL_NAME = "/home/rwkv/rwkv7-g1f-7.2b-20260414-ctx8192"
DEFAULT_PROMPT = "User: Write HTML: Lex Fridman Podcast, in diverse and elegant styles\n\nAssistant: <think"
TITLE_MODEL_NAME = "RWKV-7 7.2B"
TITLE_PRECISION = "FP16"
TITLE_GPU_NAME = "RTX 5090"
WINDOW_TITLE = "RWKV-7 batch demo4"


@dataclass
class ParsedPage:
    thinking_text: str
    answer_text: str
    html_text: str
    render_html: str
    stage: str
    page_render: bool


def put_drop(out_q, msg):
    try:
        out_q.put_nowait(msg)
        return False
    except queue.Full:
        return True


def put_reliable(out_q, msg, shutdown_event=None, timeout=0.1):
    while shutdown_event is None or not shutdown_event.is_set():
        try:
            out_q.put(msg, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def make_model_args(model_name):
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = model_name
    return args


def load_prompt(cli):
    if cli.prompt:
        return cli.prompt.replace("\\n", "\n")
    if cli.prompt_file:
        return Path(cli.prompt_file).read_text(encoding="utf-8")
    return DEFAULT_PROMPT


def wrap_partial_html(fragment, caption="HTML is still streaming", show_caption=True):
    escaped_caption = html.escape(caption)
    caption_html = f'<div class="pending"><h1>{escaped_caption}</h1></div>' if show_caption else ""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{ margin: 0; min-height: 100%; font-family: Inter, system-ui, sans-serif; }}
    body {{ background: #f6f4ef; color: #202020; }}
    .pending {{ padding: 24px; border: 2px dashed #999; margin: 18px; border-radius: 12px; }}
    .pending h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .raw-output {{ margin: 0; padding: 18px 20px 28px; white-space: pre-wrap; overflow-wrap: anywhere; font: 24px/1.0 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  {caption_html}
  {fragment}
</body>
</html>"""


def render_raw_text_preview(raw):
    text = compact_stream_text("<think" + raw, 9000)
    fragment = f'<pre class="raw-output">{html.escape(text)}</pre>'
    return wrap_partial_html(fragment, show_caption=False)


def compact_stream_text(text, limit=6000):
    text = text.strip()
    if len(text) <= limit:
        return text
    keep_head = max(800, limit // 4)
    keep_tail = max(1200, limit - keep_head)
    omitted = len(text) - keep_head - keep_tail
    return text[:keep_head] + f"\n\n... omitted {omitted} characters ...\n\n" + text[-keep_tail:]


def render_html_source_preview(raw, html_text):
    fragment = f'<pre class="raw-output">{html.escape(compact_stream_text(raw, 9000))}</pre>'
    return wrap_partial_html(fragment, show_caption=False)


def html_has_closed_document(html_text):
    lower = html_text.lower()
    return "</html>" in lower or "</body>" in lower


def html_visual_ready(html_text):
    lower = html_text.lower()
    body_match = re.search(r"<body\b[^>]*>", lower)
    if not body_match:
        return html_has_closed_document(html_text)
    after_body = html_text[body_match.end() :]
    if len(after_body.strip()) < 80 and not html_has_closed_document(html_text):
        return False
    return re.search(r"<(header|main|section|article|nav|div|h1|h2|p|ul|ol|img|button|footer)\b", after_body, re.I) is not None


FENCE_RE = re.compile(r"```[ \t]*([a-zA-Z0-9_-]*)[^\S\r\n]*(?:\r?\n|$)")
CLOSING_FENCE_RE = re.compile(r"(^|\r?\n)```[ \t]*(?=\r?\n|$)")
OPENING_FENCE_LINE_RE = re.compile(r"```[ \t]*([a-zA-Z0-9_-]*)[^\S\r\n]*$")
ONLY_CLOSING_FENCE_LINE_RE = re.compile(r"^[ \t]*```[ \t]*(?:\r?\n)?$")
MARKER_BACKTRACK = 32


def find_closing_fence(text, start):
    match = CLOSING_FENCE_RE.search(text, start)
    if not match:
        return None
    return match.start() + len(match.group(1)), match.end()


def find_html_candidate(text):
    for match in FENCE_RE.finditer(text):
        content_start = match.end()
        close = find_closing_fence(text, content_start)
        content = text[content_start:] if close is None else text[content_start : close[0]]
        lang = match.group(1).lower()
        if lang == "html" or "<html" in content.lower() or "<!doctype" in content.lower():
            return content, match.start()

    lower = text.lower()
    starts = [pos for pos in (lower.find("<!doctype"), lower.find("<html")) if pos >= 0]
    if starts:
        start = min(starts)
        return text[start:], start
    return "", None


@dataclass
class HtmlCompletionScanner:
    scan_pos: int = 0
    marker_scan_pos: int = 0
    in_fence: bool = False
    html_candidate: bool = False
    html_started: bool = False
    think_closed: bool = False
    content_offset: int = 0

    def update(self, text):
        if not self.think_closed:
            close = text.find("</think>")
            if close < 0:
                return None
            self.think_closed = True
            self.content_offset = close + len("</think>")
            self.scan_pos = 0
            self.marker_scan_pos = 0
            self.in_fence = False
            self.html_candidate = False
            self.html_started = False

        content = text[self.content_offset :]
        frozen_content = self._update_html_content(content)
        if frozen_content is None:
            return None
        return text[: self.content_offset + len(frozen_content)]

    def _update_html_content(self, text):
        direct_end = self._scan_direct_markers(text)
        if direct_end is not None:
            return text[:direct_end]

        while True:
            line_end = text.find("\n", self.scan_pos)
            if line_end < 0:
                break
            end = line_end + 1
            line = text[self.scan_pos:end]
            frozen = self._scan_line(text, line, end)
            if frozen is not None:
                return frozen
            self.scan_pos = end

        trailing = text[self.scan_pos :]
        if self.in_fence and trailing:
            if not self.html_candidate and self._line_has_html_start(trailing):
                self.html_candidate = True
                self.html_started = True
            if self.html_candidate and ONLY_CLOSING_FENCE_LINE_RE.match(trailing):
                return text
        return None

    def _scan_line(self, full_text, line, end):
        if not self.in_fence:
            match = OPENING_FENCE_LINE_RE.search(line.rstrip("\r\n"))
            if match:
                self.in_fence = True
                self.html_candidate = match.group(1).lower() == "html"
                self.html_started = self.html_started or self.html_candidate
            return None

        if not self.html_candidate and self._line_has_html_start(line):
            self.html_candidate = True
            self.html_started = True
        if ONLY_CLOSING_FENCE_LINE_RE.match(line):
            if self.html_candidate:
                return full_text[:end]
            self.in_fence = False
            self.html_candidate = False
        return None

    @staticmethod
    def _line_has_html_start(line):
        lower = line.lower()
        return "<html" in lower or "<!doctype" in lower

    def _scan_direct_markers(self, text):
        start = max(0, self.marker_scan_pos - MARKER_BACKTRACK)
        lower = text[start:].lower()
        if not self.html_started and ("<html" in lower or "<!doctype html" in lower):
            self.html_started = True
        close = lower.find("</html>") if self.html_started else -1
        self.marker_scan_pos = max(0, len(text) - MARKER_BACKTRACK)
        if close >= 0:
            return start + close + len("</html>")
        return None


def open_jsonl_log(cfg):
    log_path = cfg.get("jsonl_log")
    if not log_path:
        return None, None
    path = Path(log_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.open("w", encoding="utf-8", buffering=1)


def write_jsonl_line(log_file, text):
    if log_file is None:
        return
    log_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def write_unlogged_jsonl(log_file, raw_pages, logged_pages):
    if log_file is None:
        return
    for idx, text in enumerate(raw_pages):
        if not logged_pages[idx]:
            write_jsonl_line(log_file, text)
            logged_pages[idx] = True


def parse_generated_page(raw):
    assistant_text = "<think" + raw
    body = assistant_text[len("<think") :]
    if body.startswith(">"):
        body = body[1:]

    close = body.find("</think>")
    if close < 0:
        thinking = body
        render = render_raw_text_preview(raw)
        return ParsedPage(thinking, "", "", render, "think", False)

    thinking = body[:close]
    after = body[close + len("</think>") :]
    html_text, html_start = find_html_candidate(after)
    if html_start is None:
        answer = after.strip()
    else:
        answer = after[:html_start].strip()

    if html_text.strip():
        page_render = html_visual_ready(html_text)
        if page_render:
            render_html = html_text.strip()
            if "<html" not in render_html.lower() and "<!doctype" not in render_html.lower():
                render_html = wrap_partial_html(render_html, "Partial HTML")
            stage = "html"
        else:
            render_html = render_html_source_preview(raw, html_text)
            stage = "html"
    elif answer:
        render_html = render_raw_text_preview(raw)
        stage = "answer"
        page_render = False
    else:
        render_html = render_raw_text_preview(raw)
        stage = "answer"
        page_render = False

    return ParsedPage(thinking, answer, html_text, render_html, stage, page_render)


def build_host_html(cols, rows, page_scale, allow_scripts, group_count, initial_prompt):
    frame_count = cols * rows
    sandbox = "allow-same-origin allow-scripts" if allow_scripts else "allow-same-origin"
    cells = []
    for idx in range(frame_count):
        cells.append(
            f"""
      <section class="cell" id="cell-{idx}">
        <div class="viewport">
          <iframe class="page-frame active" id="frame-{idx}-0" sandbox="{sandbox}" scrolling="yes"></iframe>
          <iframe class="page-frame hidden" id="frame-{idx}-1" sandbox="{sandbox}" scrolling="yes"></iframe>
        </div>
        <div class="caption" id="caption-{idx}">#{idx + 1} | pending | 0 bytes</div>
      </section>"""
        )
    cells_html = "\n".join(cells)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    :root {{ --page-scale: {page_scale}; }}
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }}
    body {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #grid {{
      box-sizing: border-box; height: 100vh; padding: 1px; background: #000;
      display: grid; grid-template-columns: repeat({cols}, 1fr); grid-template-rows: repeat({rows}, 1fr);
      gap: 1px;
    }}
    .cell {{ position: relative; min-width: 0; min-height: 0; background: #11151d; border: 0; overflow: hidden; }}
    .viewport {{ position: absolute; inset: 0; overflow: hidden; background: #f8f6ef; }}
    .cell.blanked .viewport {{ background: #fff; }}
    .cell.blanked .page-frame {{ visibility: hidden !important; opacity: 0 !important; pointer-events: none; }}
    .page-frame {{
      position: absolute; left: 0; top: 0;
      width: calc(100% / var(--page-scale)); height: calc(100% / var(--page-scale));
      transform: scale(var(--page-scale)); transform-origin: top left; border: 0; background: white;
      overflow: auto; scrollbar-gutter: stable;
    }}
    .page-frame.active {{ visibility: visible; opacity: 1; }}
    .page-frame.hidden {{ visibility: hidden; opacity: 0; pointer-events: none; }}
    .caption {{
      position: absolute; left: 0; right: 0; bottom: 0; box-sizing: border-box;
      padding: 3px 6px; color: #d6d9df; background: rgba(0,0,0,.62);
      font: 11px/15px ui-monospace, monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      pointer-events: none;
    }}
    .cell:hover .caption {{ white-space: normal; max-height: 46%; overflow: auto; pointer-events: auto; }}
    #pager {{
      position: fixed; right: 12px; top: 12px; z-index: 20; display: flex; align-items: center; gap: 7px;
      padding: 5px 7px; color: #f2f2f2; background: rgba(0,0,0,.58); border: 1px solid rgba(255,255,255,.28);
      border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 18px/24px ui-monospace, monospace; pointer-events: auto;
    }}
    #pager button {{
      display: grid; place-items: center; text-decoration: none;
      width: 54px; height: 33px; padding: 0; color: #fff; background: rgba(255,255,255,.14);
      border: 1px solid rgba(255,255,255,.30); border-radius: 8px; cursor: pointer;
      font: 700 21px/27px ui-monospace, monospace;
    }}
    #pager button:hover {{ background: rgba(255,255,255,.22); }}
    #pager button:disabled {{ opacity: .55; cursor: default; }}
    #groupLabel {{ min-width: 42px; text-align: center; }}
    #restartButton {{
      position: fixed; right: 12px; bottom: 12px; z-index: 20;
      display: grid; place-items: center;
      width: 48px; height: 48px; padding: 0;
      color: #fff; background: rgba(0,0,0,.58);
      border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
      box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 700 25px/25px ui-monospace, monospace; cursor: pointer;
      pointer-events: auto;
    }}
    #restartButton:hover {{ background: rgba(255,255,255,.18); }}
    #restartButton:disabled {{ opacity: .55; cursor: default; }}
    #promptButton {{
      position: fixed; left: 12px; top: 12px; z-index: 20;
      display: grid; place-items: center;
      width: 48px; height: 48px; padding: 0;
      color: #fff; background: rgba(0,0,0,.58);
      border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
      box-shadow: 0 8px 24px rgba(0,0,0,.24);
      font: 700 22px/22px ui-monospace, monospace; cursor: pointer;
      pointer-events: auto;
    }}
    #promptButton:hover {{ background: rgba(255,255,255,.18); }}
    #promptButton:disabled {{ opacity: .55; cursor: default; }}
    #promptOverlay {{
      position: fixed; inset: 0; z-index: 30; display: none;
      align-items: center; justify-content: center;
      background: rgba(0,0,0,.38); pointer-events: auto;
    }}
    #promptOverlay.open {{ display: flex; }}
    #promptDialog {{
      box-sizing: border-box; width: min(900px, calc(100vw - 64px)); height: min(560px, calc(100vh - 64px));
      display: grid; grid-template-rows: auto 1fr auto; gap: 12px;
      padding: 16px; color: #f4f4f4; background: rgba(18,20,24,.94);
      border: 1px solid rgba(255,255,255,.24); border-radius: 12px;
      box-shadow: 0 18px 50px rgba(0,0,0,.40);
      font: 14px/20px ui-monospace, monospace;
    }}
    #promptDialog h2 {{ margin: 0; font: 700 18px/24px ui-monospace, monospace; }}
    #promptText {{
      box-sizing: border-box; width: 100%; height: 100%; resize: none;
      padding: 12px; color: #111; background: #fff;
      border: 1px solid rgba(255,255,255,.35); border-radius: 8px;
      font: 14px/20px ui-monospace, monospace;
    }}
    #promptActions {{ display: flex; justify-content: flex-end; gap: 10px; }}
    #promptActions button {{
      min-width: 88px; height: 36px; padding: 0 14px;
      color: #fff; background: rgba(255,255,255,.14);
      border: 1px solid rgba(255,255,255,.30); border-radius: 8px;
      font: 700 14px/20px ui-monospace, monospace; cursor: pointer;
    }}
    #promptActions button:hover {{ background: rgba(255,255,255,.22); }}
  </style>
</head>
<body>
  <main id="grid">
    {cells_html}
  </main>
  <nav id="pager">
    <button id="pagerPrev" type="button" onclick="changeGroup('prev')" disabled>&lt;</button>
    <span id="groupLabel">1/{group_count}</span>
    <button id="pagerNext" type="button" onclick="changeGroup('next')" disabled>&gt;</button>
  </nav>
  <button id="promptButton" type="button" onclick="openPromptDialog()" title="Edit prompt" aria-label="Edit prompt" disabled>✏️</button>
  <button id="restartButton" type="button" onclick="restartGeneration()" title="Restart" aria-label="Restart" disabled>🔄</button>
  <div id="promptOverlay">
    <section id="promptDialog" role="dialog" aria-modal="true" aria-label="Edit prompt">
      <h2>Edit Prompt</h2>
      <textarea id="promptText" spellcheck="false"></textarea>
      <div id="promptActions">
        <button type="button" onclick="closePromptDialog()">Cancel</button>
        <button type="button" onclick="applyPromptDialog()">OK</button>
      </div>
    </section>
  </div>
  <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
  <script>
    let currentPrompt = {json.dumps(initial_prompt, ensure_ascii=False)};
    window.setScale = function(scale) {{
      document.documentElement.style.setProperty('--page-scale', String(scale));
    }};
    let demo4Bridge = null;
    function enablePager(enabled) {{
      const prev = document.getElementById('pagerPrev');
      const next = document.getElementById('pagerNext');
      if (prev) prev.disabled = !enabled;
      if (next) next.disabled = !enabled;
      const restart = document.getElementById('restartButton');
      if (restart) restart.disabled = !enabled;
      const prompt = document.getElementById('promptButton');
      if (prompt) prompt.disabled = !enabled;
    }}
    function changeGroup(action) {{
      if (demo4Bridge) demo4Bridge.handleGroup(action);
    }}
    function restartGeneration() {{
      if (demo4Bridge) demo4Bridge.handleRestart();
    }}
    function openPromptDialog() {{
      const overlay = document.getElementById('promptOverlay');
      const text = document.getElementById('promptText');
      if (!overlay || !text) return;
      text.value = currentPrompt;
      overlay.classList.add('open');
      if (demo4Bridge) demo4Bridge.handlePromptEditor(true);
      setTimeout(function() {{ text.focus(); }}, 0);
    }}
    function closePromptDialog(notify) {{
      const overlay = document.getElementById('promptOverlay');
      if (overlay) overlay.classList.remove('open');
      if (notify !== false && demo4Bridge) demo4Bridge.handlePromptEditor(false);
    }}
    function applyPromptDialog() {{
      const text = document.getElementById('promptText');
      if (!text) return;
      const nextPrompt = text.value;
      if (nextPrompt !== currentPrompt) {{
        closePromptDialog(false);
        currentPrompt = nextPrompt;
        if (demo4Bridge) demo4Bridge.handlePrompt(nextPrompt);
      }} else {{
        closePromptDialog(true);
      }}
    }}
    window.setPrompt = function(prompt) {{
      currentPrompt = String(prompt || '');
      const text = document.getElementById('promptText');
      if (text && document.getElementById('promptOverlay').classList.contains('open')) {{
        text.value = currentPrompt;
      }}
    }};
    if (window.qt && window.qt.webChannelTransport) {{
      new QWebChannel(window.qt.webChannelTransport, function(channel) {{
        demo4Bridge = channel.objects.demo4Bridge;
        enablePager(true);
      }});
    }}
    let viewEpoch = 0;
    const pageState = Array.from({{ length: {frame_count} }}, () => ({{
      active: 0,
      loading: false,
      loadToken: 0,
      loadTimer: null,
      currentHtml: '',
      pendingItem: null
    }}));
    function frameFor(index, slot) {{
      return document.getElementById('frame-' + index + '-' + slot);
    }}
    function captionFor(index) {{
      return document.getElementById('caption-' + index);
    }}
    function cellFor(index) {{
      return document.getElementById('cell-' + index);
    }}
    function setCaption(item) {{
      const caption = captionFor(item.index);
      if (!caption) return;
      const pageNumber = item.globalIndex == null ? item.index + 1 : item.globalIndex + 1;
      const parts = [
        '#' + pageNumber,
        item.captionStage || item.stage || 'pending',
        (item.totalBytes || 0) + ' bytes'
      ];
      caption.textContent = parts.join(' | ');
    }}
    function isTextStage(item) {{
      return item && item.autoScroll === true;
    }}
    function scrollFrameToBottom(frame) {{
      try {{
        const doc = frame.contentDocument;
        const win = frame.contentWindow;
        if (!doc || !win) return false;
        const body = doc.body;
        const root = doc.documentElement;
        const height = Math.max(
          body ? body.scrollHeight : 0,
          root ? root.scrollHeight : 0
        );
        if (body) body.scrollTop = height;
        if (root) root.scrollTop = height;
        win.scrollTo(0, height);
        return true;
      }} catch (err) {{
      }}
      return false;
    }}
    function frameReady(frame) {{
      try {{
        const doc = frame.contentDocument;
        return !!(doc && doc.readyState !== 'loading' && doc.body);
      }} catch (err) {{
        return false;
      }}
    }}
    function keepTextFrameAtBottom(frame, item, state, token) {{
      if (!isTextStage(item)) return;
      const delays = [0, 16, 40, 90, 180, 320];
      for (const delay of delays) {{
        setTimeout(function() {{
          if (token !== state.loadToken) return;
          if (item.epoch !== undefined && item.epoch !== viewEpoch) return;
          if (frameFor(item.index, state.active) !== frame) return;
          scrollFrameToBottom(frame);
        }}, delay);
      }}
    }}
    window.setPageCaptions = function(batch) {{
      for (const item of batch) setCaption(item);
    }};
    window.setGroup = function(current, total) {{
      const label = document.getElementById('groupLabel');
      if (label) label.textContent = String(current + 1) + '/' + String(total);
    }};
    window.resetFrames = function(captions, epoch) {{
      if (epoch !== undefined && epoch !== null) viewEpoch = epoch;
      for (let index = 0; index < pageState.length; index++) {{
        const state = pageState[index];
        state.loadToken += 1;
        if (state.loadTimer) clearTimeout(state.loadTimer);
        state.loadTimer = null;
        state.loading = false;
        state.currentHtml = '';
        state.pendingItem = null;
        const cell = cellFor(index);
        if (cell) cell.classList.add('blanked');
        for (let slot = 0; slot < 2; slot++) {{
          const frame = frameFor(index, slot);
          if (!frame) continue;
          frame.onload = null;
        }}
      }}
      for (const item of captions || []) setCaption(item);
    }};
    function startBufferedLoad(item) {{
      if (item.epoch !== undefined && item.epoch !== viewEpoch) return;
      setCaption(item);
      if (typeof item.html !== 'string') {{
        return;
      }}
      const state = pageState[item.index];
      if (!state || item.html === state.currentHtml) {{
        return;
      }}
      if (state.loading) {{
        state.pendingItem = item;
        return;
      }}
      const nextSlot = 1 - state.active;
      const hidden = frameFor(item.index, nextSlot);
      const visible = frameFor(item.index, state.active);
      if (!hidden || !visible) return;
      state.loading = true;
      state.loadToken += 1;
      const token = state.loadToken;
      if (state.loadTimer) clearTimeout(state.loadTimer);
      state.pendingItem = null;
      const finishLoad = function(fromTimeout) {{
        if (token !== state.loadToken) return;
        if (item.epoch !== undefined && item.epoch !== viewEpoch) return;
        if (fromTimeout && isTextStage(item) && !frameReady(hidden)) {{
          state.loadTimer = setTimeout(function() {{ finishLoad(true); }}, 80);
          return;
        }}
        if (state.loadTimer) {{
          clearTimeout(state.loadTimer);
          state.loadTimer = null;
        }}
        hidden.onload = null;
        if (isTextStage(item)) scrollFrameToBottom(hidden);
        visible.classList.remove('active');
        visible.classList.add('hidden');
        hidden.classList.remove('hidden');
        hidden.classList.add('active');
        state.active = nextSlot;
        state.currentHtml = item.html;
        state.loading = false;
        setCaption(item);
        const cell = cellFor(item.index);
        if (cell) cell.classList.remove('blanked');
        keepTextFrameAtBottom(hidden, item, state, token);
        if (state.pendingItem && state.pendingItem.html !== state.currentHtml) {{
          const pending = state.pendingItem;
          state.pendingItem = null;
          startBufferedLoad(pending);
        }}
      }};
      hidden.onload = function() {{ finishLoad(false); }};
      state.loadTimer = setTimeout(function() {{ finishLoad(true); }}, isTextStage(item) ? 700 : 1200);
      hidden.srcdoc = item.html;
    }}
    window.updatePages = function(batch) {{
      for (const item of batch) {{
        if (item.epoch !== undefined && item.epoch !== viewEpoch) continue;
        const caption = document.getElementById('caption-' + item.index);
        if (!caption) continue;
        startBufferedLoad(item);
      }}
    }};
  </script>
</body>
</html>"""


def snapshot_dirty(out_q, raw_pages, dirty, sent_lengths, finished_pages, dropped, force=False):
    if not dirty and not force:
        return dropped
    updates = []
    sent_indices = []
    for idx in sorted(dirty):
        raw = raw_pages[idx]
        start = sent_lengths[idx]
        finished = finished_pages[idx]
        if len(raw) <= start and not finished:
            sent_indices.append(idx)
            continue
        updates.append(
            {
                "index": idx,
                "delta": raw[start:],
                "finished": finished,
                "totalBytes": len(raw.encode("utf-8")) if finished else None,
            }
        )
        sent_indices.append(idx)
    if not updates:
        dirty.difference_update(sent_indices)
        return dropped
    if put_drop(out_q, ("pages_delta", updates)):
        dropped += 1
        return dropped
    for idx in sent_indices:
        sent_lengths[idx] = len(raw_pages[idx])
    dirty.difference_update(sent_indices)
    return dropped


def send_finished_captions(out_q, raw_pages):
    payload = [
        {
            "index": idx,
            "captionStage": "finish",
            "totalBytes": len(text.encode("utf-8")),
        }
        for idx, text in enumerate(raw_pages)
    ]
    put_drop(out_q, ("pages_finished", payload))


def send_page_finished(out_q, idx, text):
    msg = (
        "page_finished",
        {
            "index": idx,
            "text": text,
            "captionStage": "finish",
            "totalBytes": len(text.encode("utf-8")),
        },
    )
    try:
        out_q.put(msg, timeout=0.02)
        return False
    except queue.Full:
        return put_drop(out_q, msg)


def model_process(out_q, ctrl_q, shutdown_event, cfg):
    model_dir = Path(cfg["model_dir"]).resolve()
    sys.path.insert(0, str(model_dir))

    import torch
    from reference.rwkv7 import RWKV_x070
    from reference.utils import TRIE_TOKENIZER, sampler_top_p_fast

    page_count = cfg["cols"] * cfg["rows"] * cfg["groups"]
    current_prompt = cfg["prompt"]
    flush_dt = 1.0 / max(1, cfg["producer_flush_hz"])
    perf_interval = max(1, cfg["perf_interval"])
    perf_sync_interval = max(0, cfg["perf_sync_interval"])
    sampler_top_p = cfg["sampler_top_p"]
    sampler_temp = cfg["sampler_temp"]
    sampler_top_k = cfg["sampler_top_k"]
    presence_penalty = cfg["presence_penalty"]
    generation_length = cfg["generation_length"]

    if cfg.get("model_nice", 0) > 0:
        try:
            os.nice(int(cfg["model_nice"]))
        except OSError:
            pass

    state = None
    raw_pages = []
    logged_pages = []
    log_file = None
    dropped = 0
    try:
        model = RWKV_x070(make_model_args(cfg["model_name"]))
        tokenizer = TRIE_TOKENIZER(str(model_dir / "reference" / "rwkv_vocab_v20230424.txt"))
        generation_id = 0
        seed_base = int(time.time_ns() & 0x7FFFFFFF)
        baseline_prompt = None
        baseline_state = None
        baseline_out = None

        def clone_tensor_list(items):
            return [item.detach().clone() if torch.is_tensor(item) else item for item in items]

        def compare_tensor_lists(lhs, rhs, label):
            if lhs is None or rhs is None:
                return True, f"{label}: missing baseline"
            if len(lhs) != len(rhs):
                return False, f"{label}: len {len(lhs)} != {len(rhs)}"
            for idx, (a, b) in enumerate(zip(lhs, rhs)):
                if torch.is_tensor(a) or torch.is_tensor(b):
                    if not (torch.is_tensor(a) and torch.is_tensor(b)):
                        return False, f"{label}[{idx}]: tensor/type mismatch"
                    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                        return False, f"{label}[{idx}]: meta {tuple(a.shape)} {a.dtype} {a.device} != {tuple(b.shape)} {b.dtype} {b.device}"
                    if not torch.equal(a, b):
                        max_diff = (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()
                        return False, f"{label}[{idx}]: value mismatch max_abs_diff={max_diff:.6g}"
                elif a != b:
                    return False, f"{label}[{idx}]: value mismatch"
            return True, f"{label}: exact match"

        def compare_tensor(lhs, rhs, label):
            if lhs is None or rhs is None:
                return True, f"{label}: missing baseline"
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.device != rhs.device:
                return False, f"{label}: meta {tuple(lhs.shape)} {lhs.dtype} {lhs.device} != {tuple(rhs.shape)} {rhs.dtype} {rhs.device}"
            if torch.equal(lhs, rhs):
                return True, f"{label}: exact match"
            max_diff = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs().max().item()
            return False, f"{label}: value mismatch max_abs_diff={max_diff:.6g}"

        while not shutdown_event.is_set():
            if state is not None:
                del state
                torch.cuda.empty_cache()
            if log_file is not None:
                log_file.close()
                log_file = None

            generation_id += 1
            seed = seed_base + generation_id
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            state = model.generate_zero_state(page_count)
            raw_pages = ["" for _ in range(page_count)]
            sent_lengths = [0 for _ in range(page_count)]
            finished_pages = [False for _ in range(page_count)]
            logged_pages = [False for _ in range(page_count)]
            completion_scanners = [HtmlCompletionScanner() for _ in range(page_count)]
            log_path, log_file = open_jsonl_log(cfg)
            dirty = set()
            put_reliable(out_q, ("clear_all", None), shutdown_event)
            put_drop(out_q, ("status", "Loading prompts..."))
            if log_path is not None:
                put_drop(out_q, ("status", f"Writing JSONL log to {log_path}"))

            prompts = [current_prompt for _ in range(page_count)]
            out = model.forward_batch([tokenizer.encode(prompt) for prompt in prompts], state)
            if baseline_state is None or baseline_prompt != current_prompt:
                baseline_prompt = current_prompt
                baseline_state = clone_tensor_list(state)
                baseline_out = out.detach().clone()
                put_drop(out_q, ("prefill_check", {"ok": True, "message": f"Prefill baseline captured | generation {generation_id}"}))
            else:
                state_ok, state_msg = compare_tensor_lists(state, baseline_state, "prefill state")
                out_ok, out_msg = compare_tensor(out, baseline_out, "prefill logits")
                if state_ok and out_ok:
                    put_drop(out_q, ("prefill_check", {"ok": True, "message": f"Prefill check OK | generation {generation_id}"}))
                else:
                    put_reliable(
                        out_q,
                        (
                            "prefill_check",
                            {
                                "ok": False,
                                "message": f"Prefill check FAILED | generation {generation_id} | {state_msg} | {out_msg}",
                            },
                        ),
                        shutdown_event,
                    )
            token_presence = None
            if presence_penalty != 0.0:
                token_presence = torch.zeros(
                    (page_count, out.shape[-1]),
                    dtype=torch.bool,
                    device=out.device,
                )
            decoders = [codecs.getincrementaldecoder("utf-8")("strict") for _ in range(page_count)]
            last_flush = time.perf_counter()
            perf_start = last_flush
            perf_tokens = 0
            stop_requested = False
            restart_requested = False
            paused = False

            def handle_control_message(msg):
                nonlocal baseline_out
                nonlocal baseline_prompt
                nonlocal baseline_state
                nonlocal current_prompt
                nonlocal paused
                nonlocal restart_requested
                nonlocal stop_requested
                if msg == "stop":
                    stop_requested = True
                elif msg == "restart":
                    restart_requested = True
                elif msg == "pause":
                    paused = True
                elif msg == "resume":
                    paused = False
                elif isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "prompt":
                    current_prompt = str(msg[1])
                    baseline_prompt = None
                    baseline_state = None
                    baseline_out = None
                    restart_requested = True

            step = 0
            while generation_length <= 0 or step < generation_length:
                if shutdown_event.is_set():
                    break

                try:
                    while True:
                        handle_control_message(ctrl_q.get_nowait())
                        if stop_requested or restart_requested:
                            break
                except queue.Empty:
                    pass
                if stop_requested:
                    put_drop(out_q, ("status", "Stopping model worker"))
                    break
                if restart_requested:
                    put_drop(out_q, ("status", "Restarting generation"))
                    break
                if paused:
                    put_drop(out_q, ("status", "Generation paused"))
                    while paused and not stop_requested and not restart_requested and not shutdown_event.is_set():
                        try:
                            handle_control_message(ctrl_q.get(timeout=0.1))
                        except queue.Empty:
                            continue
                    if stop_requested:
                        put_drop(out_q, ("status", "Stopping model worker"))
                        break
                    if restart_requested:
                        put_drop(out_q, ("status", "Restarting generation"))
                        break
                    if shutdown_event.is_set():
                        break
                    put_drop(out_q, ("status", "Generation resumed"))

                new_tokens_tensor = sampler_top_p_fast(
                    out,
                    sampler_top_p,
                    sampler_temp,
                    sampler_top_k,
                    token_presence,
                    presence_penalty,
                )
                flat_tokens = new_tokens_tensor.view(-1).detach().cpu().tolist()

                if hasattr(model, "forward_seq_batch_1"):
                    out = model.forward_seq_batch_1(new_tokens_tensor, state, False)
                else:
                    out = model.forward_batch([[int(x)] for x in flat_tokens], state)
                perf_tokens += page_count

                for idx, token_id in enumerate(flat_tokens):
                    text = decoders[idx].decode(tokenizer.idx2token[token_id], final=False)
                    if text and not finished_pages[idx]:
                        raw_pages[idx] += text
                        frozen = completion_scanners[idx].update(raw_pages[idx])
                        if frozen is not None:
                            raw_pages[idx] = frozen
                            finished_pages[idx] = True
                            write_jsonl_line(log_file, frozen)
                            logged_pages[idx] = True
                            if send_page_finished(out_q, idx, frozen):
                                dirty.add(idx)
                            else:
                                sent_lengths[idx] = len(frozen)
                                dirty.discard(idx)
                        else:
                            dirty.add(idx)

                now = time.perf_counter()
                if now - last_flush >= flush_dt:
                    dropped = snapshot_dirty(out_q, raw_pages, dirty, sent_lengths, finished_pages, dropped)
                    last_flush = now

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
                            "status",
                            f"{TITLE_MODEL_NAME} {TITLE_PRECISION} bsz{page_count} @ {TITLE_GPU_NAME} | "
                            f"Token/s {tps}",
                        ),
                    )
                if all(finished_pages):
                    put_drop(out_q, ("status", f"All {page_count} pages completed"))
                    break
                step += 1

            if stop_requested:
                return
            if restart_requested:
                continue
            dropped = snapshot_dirty(out_q, raw_pages, dirty, sent_lengths, finished_pages, dropped, force=True)
            write_unlogged_jsonl(log_file, raw_pages, logged_pages)
            completed = sum(finished_pages)
            if completed == page_count:
                send_finished_captions(out_q, raw_pages)
                put_drop(out_q, ("status", f"Generation finished: all {page_count} pages completed"))
            elif shutdown_event.is_set():
                put_drop(out_q, ("status", f"Generation stopped: {completed}/{page_count} pages completed"))
            else:
                put_drop(out_q, ("status", f"Token budget reached: {completed}/{page_count} pages completed"))
            if log_file is not None:
                log_file.close()
                log_file = None
            while not shutdown_event.is_set():
                try:
                    msg = ctrl_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if msg == "stop":
                    return
                if msg == "restart":
                    break
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "prompt":
                    current_prompt = str(msg[1])
                    baseline_prompt = None
                    baseline_state = None
                    baseline_out = None
                    break
            if shutdown_event.is_set():
                return
    except Exception as exc:
        put_drop(out_q, ("error", f"{type(exc).__name__}: {exc}"))
    finally:
        write_unlogged_jsonl(log_file, raw_pages, logged_pages)
        if log_file is not None:
            log_file.close()
        put_drop(out_q, ("done", None))


def gui_process(in_q, ctrl_q, shutdown_event, cfg):
    if cfg.get("gui_nice", 0) > 0:
        try:
            os.nice(int(cfg["gui_nice"]))
        except OSError:
            pass

    if cfg.get("disable_webengine_gpu", True):
        flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        if "--disable-gpu" not in flags:
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (flags + " --disable-gpu --disable-gpu-compositing").strip()

    from PySide6 import QtCore, QtWidgets
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView

    class BridgeObject(QtCore.QObject):
        def __init__(self, owner):
            super().__init__(owner)
            self.owner = owner

        @QtCore.Slot(str)
        def handleGroup(self, action):
            QtCore.QTimer.singleShot(0, lambda action=action: self.owner.handle_group_action(action))

        @QtCore.Slot()
        def handleRestart(self):
            QtCore.QTimer.singleShot(0, self.owner.request_restart)

        @QtCore.Slot(str)
        def handlePrompt(self, prompt):
            QtCore.QTimer.singleShot(0, lambda prompt=prompt: self.owner.handle_prompt(prompt))

        @QtCore.Slot(bool)
        def handlePromptEditor(self, opened):
            QtCore.QTimer.singleShot(0, lambda opened=opened: self.owner.handle_prompt_editor(opened))

    class HtmlGridView(QWebEngineView):
        def __init__(self):
            super().__init__()
            self.visible_count = cfg["cols"] * cfg["rows"]
            self.group_count = cfg["groups"]
            self.page_count = self.visible_count * self.group_count
            self.current_group = 0
            self.raw_pages = ["" for _ in range(self.page_count)]
            self.raw_bytes = [0 for _ in range(self.page_count)]
            self.byte_integrity_error = False
            self.byte_integrity_message = ""
            self.prefill_check_error = False
            self.prefill_check_message = ""
            self.dirty = set()
            self.loaded = False
            self.pending_status = "Starting demo4..."
            self.current_prompt = cfg["prompt"]
            self.current_scale = cfg["page_scale"]
            self.last_rendered_stage = ["" for _ in range(self.page_count)]
            self.last_rendered_html_bytes = [0 for _ in range(self.page_count)]
            self.last_rendered_at = [0.0 for _ in range(self.page_count)]
            self.next_render_at = [0.0 for _ in range(self.page_count)]
            self.page_phase = self.make_page_phases()
            self.render_cursor = 0
            self.force_render = set()
            self.finished_pages = [False for _ in range(self.page_count)]
            self.seen_error = False
            self.pending_group = None
            self.ignore_until_clear_all = False
            self.view_epoch = 0
            self.render_defer_until = 0.0
            self.group_render_delay_ms = max(0, cfg["group_render_delay_ms"])
            self.channel = QWebChannel(self.page())
            self.bridge_object = BridgeObject(self)
            self.channel.registerObject("demo4Bridge", self.bridge_object)
            self.page().setWebChannel(self.channel)
            self.refresh_title()
            self.resize(cfg["window_w"], cfg["window_h"])

            settings = self.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, False)

            self.loadFinished.connect(self.on_loaded)
            self.setHtml(
                build_host_html(
                    cfg["cols"],
                    cfg["rows"],
                    cfg["page_scale"],
                    cfg["allow_scripts"],
                    self.group_count,
                    self.current_prompt,
                ),
                QtCore.QUrl("about:blank"),
            )

            self.poll_timer = QtCore.QTimer(self)
            self.poll_timer.timeout.connect(self.poll_messages)
            self.poll_timer.start(max(1, cfg["ui_poll_ms"]))

            self.render_timer = QtCore.QTimer(self)
            self.render_timer.timeout.connect(self.render_dirty)
            self.render_timer.start(max(1, int(1000 / max(1, cfg["render_hz"]))))

            self.auto_group_timer = None
            auto_group_seconds = float(cfg.get("auto_switch_group_seconds", 0.0) or 0.0)
            if self.group_count > 1 and auto_group_seconds > 0.0:
                self.auto_group_timer = QtCore.QTimer(self)
                self.auto_group_timer.timeout.connect(self.auto_switch_group)
                self.auto_group_timer.start(int(max(3.0, auto_group_seconds) * 1000))

        def set_status(self, text):
            self.pending_status = text or ""
            warning = ""
            if self.byte_integrity_error:
                warning = f" | BYTE MISMATCH: {self.byte_integrity_message}"
            if self.prefill_check_error:
                warning += f" | PREFILL MISMATCH: {self.prefill_check_message}"
            title = (
                f"{WINDOW_TITLE}{warning} | finish {sum(self.finished_pages):,}/{self.page_count:,} "
                f"| bytes {sum(self.raw_bytes):,}"
            )
            if self.pending_status:
                title += " | " + self.pending_status
            self.setWindowTitle(title)

        def refresh_title(self):
            self.set_status(self.pending_status)

        def warn_byte_integrity(self, message):
            if not self.byte_integrity_error:
                self.byte_integrity_error = True
                self.byte_integrity_message = message
            self.refresh_title()

        def verify_finished_page_bytes(self, idx, source, expected_total=None):
            actual = len(self.raw_pages[idx].encode("utf-8"))
            displayed = self.raw_bytes[idx]
            mismatch = []
            if expected_total is not None and expected_total != actual:
                mismatch.append(f"msg={expected_total:,} actual={actual:,}")
            if displayed != actual:
                mismatch.append(f"display={displayed:,} actual={actual:,}")
            if mismatch:
                self.warn_byte_integrity(f"page {idx + 1} {source} " + " ".join(mismatch))
                self.raw_bytes[idx] = actual
                return True
            return False

        def verify_total_bytes(self, source):
            displayed = sum(self.raw_bytes)
            actual = sum(len(text.encode("utf-8")) for text in self.raw_pages)
            if displayed != actual:
                self.warn_byte_integrity(f"{source} total display={displayed:,} actual={actual:,}")
                return False
            return True

        def visible_start(self):
            return self.current_group * self.visible_count

        def visible_indices(self):
            start = self.visible_start()
            return range(start, min(start + self.visible_count, self.page_count))

        def is_visible(self, idx):
            start = self.visible_start()
            return start <= idx < start + self.visible_count

        def local_index(self, idx):
            return idx - self.visible_start()

        def caption_item(self, idx, stage=None):
            return {
                "index": self.local_index(idx),
                "globalIndex": idx,
                "captionStage": "finish" if self.finished_pages[idx] else stage or self.last_rendered_stage[idx] or "pending",
                "totalBytes": self.raw_bytes[idx],
            }

        def visible_caption_items(self):
            return [self.caption_item(idx) for idx in self.visible_indices()]

        def reset_visible_frames(self):
            script = (
                "window.resetFrames("
                + json.dumps(self.visible_caption_items(), ensure_ascii=False)
                + ","
                + json.dumps(self.view_epoch)
                + ");"
            )
            self.run_js(script)

        def reset_local_generation_state(self, status_text):
            self.raw_pages = ["" for _ in range(self.page_count)]
            self.raw_bytes = [0 for _ in range(self.page_count)]
            self.byte_integrity_error = False
            self.byte_integrity_message = ""
            self.dirty.clear()
            self.last_rendered_stage = ["" for _ in range(self.page_count)]
            self.last_rendered_html_bytes = [0 for _ in range(self.page_count)]
            self.last_rendered_at = [0.0 for _ in range(self.page_count)]
            self.force_render.clear()
            self.finished_pages = [False for _ in range(self.page_count)]
            self.seen_error = False
            self.reset_render_schedule()
            target_group = self.current_group if self.pending_group is None else self.pending_group
            self.current_group = max(0, min(self.group_count - 1, target_group))
            self.pending_group = None
            self.view_epoch += 1
            self.render_defer_until = 0.0
            self.run_js(
                "window.setGroup("
                + json.dumps(self.current_group)
                + ","
                + json.dumps(self.group_count)
                + ");"
            )
            self.reset_visible_frames()
            self.dirty.update(self.visible_indices())
            self.set_status(status_text)

        def request_restart(self):
            if self.ignore_until_clear_all:
                return
            try:
                ctrl_q.put_nowait("restart")
            except queue.Full:
                self.set_status("Restart request queue is full")
                return
            self.ignore_until_clear_all = True
            self.reset_local_generation_state("Restarting generation...")

        def handle_prompt(self, prompt):
            if prompt == self.current_prompt:
                return
            if self.ignore_until_clear_all:
                self.set_status("Prompt change ignored while restart is pending")
                self.run_js("window.setPrompt(" + json.dumps(self.current_prompt, ensure_ascii=False) + ");")
                return
            try:
                ctrl_q.put_nowait(("prompt", prompt))
            except queue.Full:
                self.set_status("Prompt update queue is full")
                self.run_js("window.setPrompt(" + json.dumps(self.current_prompt, ensure_ascii=False) + ");")
                try:
                    ctrl_q.put_nowait("resume")
                except queue.Full:
                    pass
                return
            self.current_prompt = prompt
            cfg["prompt"] = prompt
            self.ignore_until_clear_all = True
            self.reset_local_generation_state("Restarting generation with new prompt...")

        def handle_prompt_editor(self, opened):
            try:
                ctrl_q.put_nowait("pause" if opened else "resume")
            except queue.Full:
                self.set_status("Prompt editor control queue is full")
                return
            self.set_status("Generation paused for prompt editing" if opened else "Generation resumed")

        def switch_group(self, group):
            group = max(0, min(self.group_count - 1, group))
            if group == self.current_group:
                return
            previous_visible = set(self.visible_indices())
            self.current_group = group
            self.view_epoch += 1
            self.render_cursor = 0
            visible = set(self.visible_indices())
            self.dirty.difference_update(previous_visible)
            self.force_render.difference_update(previous_visible)
            self.dirty.update(visible)
            self.force_render.update(visible)
            now = time.perf_counter()
            self.render_defer_until = now + self.group_render_delay_ms / 1000.0
            for idx in visible:
                self.last_rendered_stage[idx] = ""
                self.last_rendered_html_bytes[idx] = 0
                self.last_rendered_at[idx] = 0.0
                self.next_render_at[idx] = self.render_defer_until
            self.run_js(
                "window.setGroup("
                + json.dumps(self.current_group)
                + ","
                + json.dumps(self.group_count)
                + ");"
            )
            self.reset_visible_frames()
            QtCore.QTimer.singleShot(self.group_render_delay_ms, self.render_dirty)

        def request_switch_group(self, group):
            group = max(0, min(self.group_count - 1, group))
            if not self.loaded:
                self.pending_group = group
                return
            self.pending_group = None
            self.switch_group(group)

        def handle_group_action(self, action):
            base_group = self.current_group if self.pending_group is None else self.pending_group
            if action == "prev":
                self.request_switch_group((base_group - 1) % self.group_count)
            elif action == "next":
                self.request_switch_group((base_group + 1) % self.group_count)
            else:
                try:
                    self.request_switch_group(int(action) - 1)
                except ValueError:
                    pass

        def auto_switch_group(self):
            if self.group_count <= 1:
                return
            base_group = self.current_group if self.pending_group is None else self.pending_group
            self.request_switch_group((base_group + 1) % self.group_count)

        def make_page_phases(self):
            if self.page_count <= 0:
                return []
            window = max(0.0, cfg["html_stagger_ms"] / 1000.0)
            if window <= 0:
                return [0.0 for _ in range(self.page_count)]
            rng = random.Random(0xD4E4F00D)
            slot = window / self.page_count
            phases = []
            for idx in range(self.page_count):
                jitter = rng.uniform(0.0, slot * 0.55)
                phases.append(idx * slot + jitter)
            return phases

        def reset_render_schedule(self, start_now=False):
            base = time.perf_counter()
            for idx in range(self.page_count):
                self.next_render_at[idx] = base if start_now else base + self.page_phase[idx]
            self.render_cursor = 0

        def on_loaded(self, ok):
            self.loaded = ok
            self.run_js(f"window.setScale({json.dumps(self.current_scale)});")
            self.run_js(
                "window.setGroup("
                + json.dumps(self.current_group)
                + ","
                + json.dumps(self.group_count)
                + ");"
            )
            self.set_status(self.pending_status)
            self.reset_render_schedule()
            self.dirty.update(self.visible_indices())
            if self.pending_group is not None:
                group = self.pending_group
                self.pending_group = None
                self.switch_group(group)
            else:
                self.reset_visible_frames()

        def run_js(self, script):
            if self.loaded:
                self.page().runJavaScript(script)

        def poll_messages(self):
            processed = 0
            while processed < cfg["max_gui_messages"]:
                try:
                    msg_type, payload = in_q.get_nowait()
                except queue.Empty:
                    break
                processed += 1
                if self.ignore_until_clear_all and msg_type not in ("clear_all", "error"):
                    continue
                if msg_type == "pages_delta":
                    caption_updates = []
                    finished_changed = False
                    bytes_changed = False
                    for item in payload:
                        idx = item.get("index")
                        if not isinstance(idx, int) or not 0 <= idx < self.page_count:
                            continue
                        delta = item.get("delta", "")
                        if delta:
                            self.raw_pages[idx] += delta
                            self.raw_bytes[idx] += len(delta.encode("utf-8"))
                            bytes_changed = True
                        if item.get("totalBytes") is not None:
                            if self.raw_bytes[idx] != item["totalBytes"]:
                                bytes_changed = True
                            self.raw_bytes[idx] = item["totalBytes"]
                        if item.get("finished"):
                            if not self.finished_pages[idx]:
                                finished_changed = True
                            self.finished_pages[idx] = True
                            self.force_render.add(idx)
                            self.next_render_at[idx] = 0.0
                            if self.verify_finished_page_bytes(idx, "pages_delta", item.get("totalBytes")):
                                bytes_changed = True
                        if self.is_visible(idx):
                            caption_updates.append(self.caption_item(idx))
                        self.dirty.add(idx)
                    if caption_updates:
                        self.run_js("window.setPageCaptions(" + json.dumps(caption_updates, ensure_ascii=False) + ");")
                    if finished_changed or bytes_changed:
                        self.refresh_title()
                elif msg_type == "page_finished":
                    idx = payload.get("index")
                    raw = payload.get("text", "")
                    if isinstance(idx, int) and 0 <= idx < self.page_count:
                        self.raw_pages[idx] = raw
                        new_bytes = payload.get("totalBytes", len(raw.encode("utf-8")))
                        bytes_changed = self.raw_bytes[idx] != new_bytes
                        self.raw_bytes[idx] = new_bytes
                        finished_changed = not self.finished_pages[idx]
                        self.finished_pages[idx] = True
                        if self.verify_finished_page_bytes(idx, "page_finished", payload.get("totalBytes")):
                            bytes_changed = True
                        self.dirty.add(idx)
                        self.force_render.add(idx)
                        self.next_render_at[idx] = 0.0
                        if self.is_visible(idx):
                            self.run_js(
                                "window.setPageCaptions("
                                + json.dumps([self.caption_item(idx)], ensure_ascii=False)
                                + ");"
                            )
                        if finished_changed or bytes_changed:
                            self.refresh_title()
                elif msg_type == "pages_finished":
                    finished_changed = False
                    bytes_changed = False
                    for item in payload:
                        idx = item.get("index")
                        if isinstance(idx, int) and 0 <= idx < self.page_count:
                            if not self.finished_pages[idx]:
                                finished_changed = True
                            self.finished_pages[idx] = True
                            if item.get("totalBytes") is not None:
                                if self.raw_bytes[idx] != item["totalBytes"]:
                                    bytes_changed = True
                                self.raw_bytes[idx] = item["totalBytes"]
                            if self.verify_finished_page_bytes(idx, "pages_finished", item.get("totalBytes")):
                                bytes_changed = True
                    self.run_js("window.setPageCaptions(" + json.dumps(self.visible_caption_items(), ensure_ascii=False) + ");")
                    self.verify_total_bytes("pages_finished")
                    if finished_changed or bytes_changed:
                        self.refresh_title()
                elif msg_type == "clear_all":
                    self.ignore_until_clear_all = False
                    self.reset_local_generation_state(self.pending_status)
                elif msg_type == "prefill_check":
                    ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
                    message = payload.get("message", "") if isinstance(payload, dict) else str(payload)
                    if ok:
                        self.prefill_check_error = False
                        self.prefill_check_message = ""
                    else:
                        self.prefill_check_error = True
                        self.prefill_check_message = message
                    self.set_status(message)
                elif msg_type == "status":
                    self.set_status(payload)
                elif msg_type == "error":
                    self.seen_error = True
                    self.set_status(f"Error: {payload}")
                elif msg_type == "done":
                    if not self.seen_error:
                        self.set_status(payload or "Done")
            if shutdown_event.is_set():
                self.close()

        def render_dirty(self):
            if not self.loaded or not self.dirty:
                return
            batch = []
            keep_dirty = set()
            now = time.perf_counter()
            if now < self.render_defer_until:
                return
            max_html_loads = max(1, cfg["max_page_loads_per_tick"])
            max_text_loads = max(1, cfg["max_text_page_loads_per_tick"])
            html_min_delta = max(0, cfg["html_min_delta"])
            html_max_stale = max(0.05, cfg["html_max_stale_ms"] / 1000.0)
            html_refresh_interval = max(0.05, cfg["html_refresh_interval_ms"] / 1000.0)
            preview_refresh_interval = max(0.05, cfg["preview_refresh_interval_ms"] / 1000.0)
            defer_interval = max(0.03, min(0.25, html_refresh_interval / 3.0))
            html_loads = 0
            text_loads = 0
            visible = list(self.visible_indices())
            indexes = visible[self.render_cursor :] + visible[: self.render_cursor]
            next_cursor = self.render_cursor
            for idx in indexes:
                if idx not in self.dirty:
                    continue
                forced = idx in self.force_render
                if not forced and now < self.next_render_at[idx]:
                    keep_dirty.add(idx)
                    continue
                parsed = parse_generated_page(self.raw_pages[idx])
                html_bytes = len(parsed.html_text)
                total_bytes = self.raw_bytes[idx]
                is_finished = self.finished_pages[idx]
                is_page_render = parsed.page_render
                if is_page_render:
                    if html_loads >= max_html_loads:
                        self.next_render_at[idx] = now + defer_interval + self.page_phase[idx] * 0.05
                        keep_dirty.add(idx)
                        continue
                elif text_loads >= max_text_loads:
                    self.next_render_at[idx] = now + preview_refresh_interval * 0.5 + self.page_phase[idx] * 0.03
                    keep_dirty.add(idx)
                    continue
                should_render = True
                if not forced and is_page_render and self.last_rendered_stage[idx] == "html":
                    byte_delta = html_bytes - self.last_rendered_html_bytes[idx]
                    stale = now - self.last_rendered_at[idx]
                    should_render = byte_delta >= html_min_delta or stale >= html_max_stale
                if not should_render:
                    self.next_render_at[idx] = now + defer_interval + self.page_phase[idx] * 0.05
                    keep_dirty.add(idx)
                    continue
                batch.append(
                    {
                        "index": self.local_index(idx),
                        "globalIndex": idx,
                        "html": parsed.render_html,
                        "stage": parsed.stage,
                        "captionStage": "finish" if is_finished else parsed.stage,
                        "totalBytes": total_bytes,
                        "autoScroll": not is_page_render,
                        "epoch": self.view_epoch,
                    }
                )
                self.last_rendered_stage[idx] = parsed.stage
                self.last_rendered_html_bytes[idx] = html_bytes
                self.last_rendered_at[idx] = now
                self.force_render.discard(idx)
                if is_page_render:
                    html_loads += 1
                    interval = html_refresh_interval
                else:
                    text_loads += 1
                    interval = preview_refresh_interval
                self.next_render_at[idx] = now + interval + self.page_phase[idx] * 0.08
                next_cursor = (self.local_index(idx) + 1) % self.visible_count
            self.dirty = keep_dirty
            self.render_cursor = next_cursor
            if batch:
                self.run_js("window.updatePages(" + json.dumps(batch, ensure_ascii=False) + ");")

    app = QtWidgets.QApplication(sys.argv[:1])
    view = HtmlGridView()
    screen = app.primaryScreen()
    if screen is not None:
        view.setGeometry(screen.availableGeometry())
    maximized = getattr(getattr(QtCore.Qt, "WindowState", QtCore.Qt), "WindowMaximized")
    view.setWindowState(view.windowState() | maximized)
    view.show()
    QtCore.QTimer.singleShot(0, view.showMaximized)
    QtCore.QTimer.singleShot(100, view.showMaximized)
    rc = app.exec()
    shutdown_event.set()
    return rc


def parse_args():
    parser = argparse.ArgumentParser(description="demo4: parallel HTML generation and QtWebEngine rendering")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--page-scale", type=float, default=0.30)
    parser.add_argument("--render-hz", type=int, default=20)
    parser.add_argument("--max-page-loads-per-tick", type=int, default=2)
    parser.add_argument("--max-text-page-loads-per-tick", type=int, default=24)
    parser.add_argument("--html-min-delta", type=int, default=2048)
    parser.add_argument("--html-max-stale-ms", type=int, default=2000)
    parser.add_argument("--html-stagger-ms", type=int, default=1000)
    parser.add_argument("--html-refresh-interval-ms", type=int, default=650)
    parser.add_argument("--preview-refresh-interval-ms", type=int, default=120)
    parser.add_argument("--producer-flush-hz", type=int, default=10)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--ui-poll-ms", type=int, default=20)
    parser.add_argument("--max-gui-messages", type=int, default=128)
    parser.add_argument("--group-render-delay-ms", type=int, default=0)
    parser.add_argument("--auto-switch-group-seconds", type=float, default=0.0, help="auto switch to the next group every N seconds; 0 disables it, positive values are clamped to at least 3 seconds")
    parser.add_argument("--allow-scripts", action="store_true")
    parser.add_argument("--enable-webengine-gpu", dest="disable_webengine_gpu", action="store_false")
    parser.set_defaults(disable_webengine_gpu=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--jsonl-log", default="demo4_outputs.jsonl", help="write one JSONL line per page, format: {\"text\":\"...\"}")
    parser.add_argument("--no-jsonl-log", dest="jsonl_log", action="store_const", const=None)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--generation-length", type=int, default=0, help="max tokens per page; 0 means run until all pages complete HTML")
    parser.add_argument("--sampler-top-p", type=float, default=0.6)
    parser.add_argument("--sampler-temp", type=float, default=1.0)
    parser.add_argument("--sampler-top-k", type=int, default=64)
    parser.add_argument("--presence-penalty", type=float, default=1.0)
    parser.add_argument("--perf-interval", type=int, default=30)
    parser.add_argument("--perf-sync-interval", type=int, default=30)
    parser.add_argument("--gui-nice", type=int, default=5)
    parser.add_argument("--model-nice", type=int, default=0)
    parser.add_argument("--window-w", type=int, default=1600)
    parser.add_argument("--window-h", type=int, default=1000)
    return parser.parse_args()


def main():
    cli = parse_args()
    cfg = vars(cli)
    cfg["prompt"] = load_prompt(cli)
    cfg["cols"] = max(1, cfg["cols"])
    cfg["rows"] = max(1, cfg["rows"])
    cfg["groups"] = max(1, cfg["groups"])
    mp.set_start_method("spawn", force=True)

    out_q = mp.Queue(maxsize=max(1, cli.queue_size))
    ctrl_q = mp.Queue(maxsize=16)
    shutdown_event = mp.Event()

    producer = mp.Process(target=model_process, args=(out_q, ctrl_q, shutdown_event, cfg), daemon=False)
    gui = mp.Process(target=gui_process, args=(out_q, ctrl_q, shutdown_event, cfg), daemon=False)
    producer.start()
    gui.start()

    try:
        producer_reported = False
        while gui.is_alive():
            if not producer.is_alive() and not producer_reported:
                producer.join(timeout=0.1)
                producer_reported = True
                if producer.exitcode == 0:
                    put_drop(out_q, ("done", "Generation process finished; window remains open."))
                    print("demo4: producer exited normally; keeping GUI open.", flush=True)
                else:
                    put_drop(out_q, ("error", f"producer exited with code {producer.exitcode}"))
                    print(f"demo4: producer exited with code {producer.exitcode}; keeping GUI open for diagnostics.", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        shutdown_event.set()
        for name, proc in (("producer", producer), ("gui", gui)):
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join()
            print(f"demo4: {name} exitcode={proc.exitcode}", flush=True)


if __name__ == "__main__":
    main()
