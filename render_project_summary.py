"""Render PROJECT_SUMMARY.qmd to PDF with a horizontal line (\\hline) after
every table row, not just under the header.

Vi sao can script rieng thay vi chi "quarto render":
    Pandoc/Quarto sinh moi bang thanh mot moi truong LaTeX `longtable` voi
    `\\toprule`/`\\midrule`/`\\bottomrule` (kieu booktabs) va dung `\\`
    thuong (khong phai `\\tabularnewline`) de ket thuc moi dong. Thu dinh
    nghia lai `\\` ngay trong `longtable` (qua `\\AtBeginEnvironment` hay
    tuong tu) se pha vo co che noi bo cua goi `longtable` (loi "Misplaced
    \\noalign") vi `\\` cung duoc longtable dung de dung "head"/"foot".

    Cach an toan: de quarto sinh file .tex binh thuong (`keep-tex: true`
    trong PROJECT_SUMMARY.qmd), roi CHEN THANG chuoi "\\hline" vao ngay sau
    moi dong du lieu trong than bang (giua `\\endlastfoot` va
    `\\end{longtable}`) bang xu ly van ban thuan, khong dung macro TeX nao
    ca - giong het nhu neu tu tay go `\\hline` vao source.

Dung:
    .venv/Scripts/python.exe render_project_summary.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
QMD_PATH = PROJECT_ROOT / "PROJECT_SUMMARY.qmd"
TEX_PATH = PROJECT_ROOT / "PROJECT_SUMMARY.tex"
PDF_PATH = PROJECT_ROOT / "PROJECT_SUMMARY.pdf"

XELATEX = Path.home() / "AppData" / "Roaming" / "TinyTeX" / "bin" / "windows" / "xelatex.exe"

ROW_BODY_PATTERN = re.compile(r"\\endlastfoot\n(.*?)\n\\end\{longtable\}", re.DOTALL)


def add_hline_after_every_row(tex_text: str) -> tuple[str, int]:
    def patch_one_table(match: re.Match) -> str:
        body_lines = match.group(1).split("\n")
        patched_lines: list[str] = []
        for line in body_lines:
            patched_lines.append(line)
            if line.rstrip().endswith("\\\\"):
                patched_lines.append("\\hline")
        return "\\endlastfoot\n" + "\n".join(patched_lines) + "\n\\end{longtable}"

    return ROW_BODY_PATTERN.subn(patch_one_table, tex_text)


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"[1/4] quarto render {QMD_PATH.name} --to pdf (sinh .tex trung gian) ...")
    run(["quarto", "render", str(QMD_PATH), "--to", "pdf"])

    if not TEX_PATH.exists():
        raise SystemExit(f"Khong thay {TEX_PATH} - kiem tra 'keep-tex: true' trong YAML cua {QMD_PATH.name}")

    print("[2/4] Chen \\hline sau moi dong du lieu trong tat ca bang ...")
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    patched_text, n_tables = add_hline_after_every_row(tex_text)
    TEX_PATH.write_text(patched_text, encoding="utf-8")
    print(f"    Da vao {n_tables} bang.")

    print("[3/4] Bien dich lai .tex da sua bang xelatex (2 lan, de TOC/tham chieu dung) ...")
    if not XELATEX.exists():
        raise SystemExit(f"Khong thay xelatex tai {XELATEX}")
    run([str(XELATEX), "-interaction=nonstopmode", str(TEX_PATH)])
    run([str(XELATEX), "-interaction=nonstopmode", str(TEX_PATH)])

    print("[4/4] Don dep file trung gian ...")
    for suffix in (".aux", ".log", ".out", ".toc"):
        stray = TEX_PATH.with_suffix(suffix)
        if stray.exists():
            stray.unlink()
    TEX_PATH.unlink(missing_ok=True)

    print(f"\nXong: {PDF_PATH}")


if __name__ == "__main__":
    main()
