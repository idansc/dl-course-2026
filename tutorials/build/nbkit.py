"""Shared notebook builder for the tutorials: one source, two notebooks
(tutorials/<stem>.ipynb with blanks for students, tutorials/solutions/<stem>.ipynb for the TA).

Inside code cells:
  #>> hint          starts a solution block; the student version gets `# TODO: hint` + `...`
  #<<               ends it
  code  # @student: replacement     the student version gets `replacement` on that line instead
Markdown wrapped in <<STUDENT>>...<</STUDENT>> or <<SOLUTION>>...<</SOLUTION>> appears in one version only.
"""
import ast
import re
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Builder:
    def __init__(self):
        self.cells = []  # (kind, source)

    def md(self, s):
        self.cells.append(("md", s.strip()))

    def code(self, s):
        self.cells.append(("code", s.strip()))

    @staticmethod
    def _code(src, student):
        out, skipping = [], False
        for line in src.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#>>"):
                indent = line[: len(line) - len(line.lstrip())]
                if student:
                    out += [f"{indent}# TODO: {stripped[3:].strip()}", f"{indent}..."]
                skipping = True
                continue
            if stripped.startswith("#<<"):
                skipping = False
                continue
            if skipping and student:
                continue
            m = re.search(r"\s*# @student:\s*(.*)$", line)
            if m:
                indent = line[: len(line) - len(line.lstrip())]
                line = indent + m.group(1) if student else line[: m.start()]
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def _md(src, student):
        keep, drop = ("STUDENT", "SOLUTION") if student else ("SOLUTION", "STUDENT")
        src = re.sub(rf"<<{drop}>>.*?<</{drop}>>\n?", "", src, flags=re.S)
        return re.sub(rf"<</?{keep}>>\n?", "", src).strip()

    def write(self, stem):
        paths = {}
        for student in (False, True):
            cells = []
            for kind, src in self.cells:
                if kind == "md":
                    cells.append(nbf.v4.new_markdown_cell(self._md(src, student)))
                else:
                    body = self._code(src, student)
                    ast.parse(body)  # both versions must at least parse
                    cells.append(nbf.v4.new_code_cell(body))
            nb = nbf.v4.new_notebook(cells=cells, metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "language_info": {"name": "python"}})
            p = ROOT / (f"{stem}.ipynb" if student else f"solutions/{stem}.ipynb")
            nbf.write(nb, p)
            paths["student" if student else "solution"] = p
        return paths
