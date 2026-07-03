from pathlib import Path
p = Path('/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_20260624/infer.py')
s = p.read_text()
start = s.index('    def _check_boxed_complete(self, text: str) -> bool:')
end = s.find('\n    def ', start + 1)
if end < 0:
    end = len(s)
new = '''    def _check_boxed_complete(self, text: str) -> bool:
        """Stop after a complete boxed answer, including box-prefix MCQ completions like A}."""
        if re.search(r"^\\s*[A-Ja-j]\\s*}", text or ""):
            return True
        k = text.find(r"\\boxed{")
        if k < 0:
            return False
        i = k + len(r"\\boxed{")
        depth = 1
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return True
            i += 1
        return False
'''
p.write_text(s[:start] + new + s[end:])
