import io
import json
import re
import signal
import sys
import traceback
import importlib
from contextlib import contextmanager
from typing import Any, Callable

_CODE_RE = re.compile(r"```repl\s*\n(.*?)(?:\n```|```)", re.DOTALL)


def find_repl_blocks(text: str) -> list[str]:
    text = text or ""
    blocks = [m.group(1).strip() for m in _CODE_RE.finditer(text)]
    if blocks:
        return blocks
    # RWKV often starts a REPL block but misses the closing fence. Treat the
    # trailing text as a block so the environment can still recover a submitted
    # answer or return a concise syntax error.
    m = re.search(r"```repl\s*\n(.*)$", text, re.DOTALL)
    return [m.group(1).strip()] if m else []


class AnswerDict(dict):
    def __init__(self):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)


class LocalRLMRepl:
    def __init__(self, context: Any, llm_query: Callable[[str], str], exec_timeout_s: float = 8.0):
        self.answer = AnswerDict()
        self.exec_timeout_s = float(exec_timeout_s)
        self.stdout_limit = 4000
        self.globals = {
            "__builtins__": self._safe_builtins(),
            "__name__": "__main__",
        }
        self.locals = {
            "context": context,
            "answer": self.answer,
            "llm_query": llm_query,
            "rlm_query": llm_query,
            "llm_query_batched": lambda prompts, model=None: [llm_query(str(p)) for p in prompts],
            "rlm_query_batched": lambda prompts, model=None: [llm_query(str(p)) for p in prompts],
            "SHOW_VARS": self.show_vars,
        }

    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        # The official RLM executes code in worker REPLs. This local trainer runs
        # REPL code in-process, so imports must not be able to terminate or mutate
        # the trainer process (os/sys/subprocess/builtins/etc.).
        allowed_roots = {
            "math", "re", "fractions", "decimal", "itertools", "functools",
            "collections", "statistics", "operator", "random", "sympy",
        }
        root = str(name).split(".", 1)[0]
        if root not in allowed_roots:
            raise ImportError(f"import of {name!r} is blocked in local RLM REPL")
        return importlib.import_module(name)

    def _safe_builtins(self):
        names = [
            "print", "len", "str", "int", "float", "list", "dict", "set", "tuple", "bool", "type",
            "isinstance", "enumerate", "zip", "map", "filter", "sorted", "reversed", "range", "min", "max",
            "sum", "abs", "round", "any", "all", "pow", "divmod", "repr", "format", "Exception", "BaseException",
            "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError", "NameError", "AssertionError",
            "ImportError", "ZeroDivisionError",
        ]
        import builtins
        safe = {k: getattr(builtins, k) for k in names if hasattr(builtins, k)}
        safe["__import__"] = self._safe_import
        safe["eval"] = None
        safe["exec"] = None
        safe["open"] = None
        safe["input"] = None
        safe["exit"] = None
        safe["quit"] = None
        return safe

    def show_vars(self) -> str:
        keys = [k for k in self.locals if not k.startswith("_") and k not in {"answer", "context"}]
        return "Available variables: " + ", ".join(keys)

    def _boxed_fallback(self, code: str) -> str | None:
        # If execution fails after the model already wrote a boxed answer, stop
        # the RLM turn instead of feeding a huge traceback back into the prompt.
        text = code or ""
        idx = text.rfind("boxed{")
        if idx < 0:
            return None
        start = idx + len("boxed{")
        depth = 1
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    inner = text[start:i]
                    if inner.strip():
                        return "\\boxed{" + inner.strip() + "}"
                    return None
            i += 1
        # Unclosed boxed expression: keep a short token tail as a best-effort answer.
        tail = text[start:start + 64].splitlines()[0].strip().strip("\'\"")
        return ("\\boxed{" + tail + "}") if tail else None

    def _short_error(self, err_text: str) -> str:
        lines = [x for x in (err_text or "").splitlines() if x.strip()]
        if not lines:
            return ""
        # Keep the exception type/message, not the generated code line dump.
        return lines[-1][:500]


    @contextmanager
    def _capture(self):
        old_out, old_err = sys.stdout, sys.stderr
        out, err = io.StringIO(), io.StringIO()
        try:
            sys.stdout, sys.stderr = out, err
            yield out, err
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def _exec_with_timeout(self, code: str, ns: dict[str, Any]):
        if self.exec_timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
            exec(code, ns, ns)  # noqa: S102
            return
        def _alarm(signum, frame):
            raise TimeoutError(f"repl block exceeded {self.exec_timeout_s:g}s")
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, self.exec_timeout_s)
        try:
            exec(code, ns, ns)  # noqa: S102
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    def execute(self, code: str) -> dict[str, Any]:
        with self._capture() as (out, err):
            try:
                ns = {**self.globals, **self.locals}
                self._exec_with_timeout(code, ns)
                for k, v in ns.items():
                    if k not in self.globals and not k.startswith("_"):
                        self.locals[k] = v
                self.locals["answer"] = self.answer
                stdout, stderr = out.getvalue(), err.getvalue()
                ok = True
            except BaseException:
                stdout = out.getvalue()
                raw_err = err.getvalue() + traceback.format_exc(limit=1)
                fallback = self._boxed_fallback(code)
                if fallback:
                    self.answer["content"] = fallback
                    self.answer["ready"] = True
                    stderr = "recovered malformed REPL with boxed fallback"
                else:
                    # Do not leak a previous turn's answer through a failed block.
                    self.answer["ready"] = False
                    stderr = self._short_error(raw_err)
                ok = False
        if len(stdout) > self.stdout_limit:
            stdout = stdout[:self.stdout_limit] + f"... +[{len(stdout)-self.stdout_limit} chars]"
        if len(stderr) > self.stdout_limit:
            stderr = stderr[:self.stdout_limit] + f"... +[{len(stderr)-self.stdout_limit} chars]"
        return {
            "ok": ok,
            "stdout": stdout,
            "stderr": stderr,
            "final_answer": str(self.answer.get("content", "")) if self.answer.get("ready") else None,
            "locals_keys": [k for k in self.locals if not k.startswith("_")],
        }


def format_repl_outputs(outputs: list[dict[str, Any]]) -> str:
    if not outputs:
        return "No REPL code block found. You must use a ```repl block."
    parts = []
    for i, r in enumerate(outputs):
        head = f"REPL output block {i+1}:" if len(outputs) > 1 else "REPL output:"
        body = []
        if r.get("stdout"):
            body.append("stdout:\n" + r["stdout"])
        if r.get("stderr"):
            body.append("stderr:\n" + r["stderr"])
        if r.get("final_answer") is not None:
            body.append("final_answer: " + str(r["final_answer"]))
        if not body:
            body.append("No output")
        parts.append(head + "\n" + "\n".join(body))
    return "\n\n".join(parts)
