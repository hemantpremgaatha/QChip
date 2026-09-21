"""Render QSCHED.md to QSCHED.pdf via markdown + headless Chrome. Needs: pip install markdown."""
import pathlib
import subprocess
import sys

import markdown

here = pathlib.Path(__file__).parent
md = (here / "QSCHED.md").read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["tables"])
css = """
@page { size: A4; margin: 22mm 20mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 13.5pt; margin-top: 18pt; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 10pt auto; font-size: 10pt; }
th, td { border: 1px solid #888; padding: 3pt 8pt; }
th { background: #eee; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
p { text-align: justify; }
"""
html = here / "QSCHED.html"
html.write_text(f"<!doctype html><meta charset='utf-8'><title>QSched</title><style>{css}</style>{body}",
                encoding="utf-8")
chrome = next((p for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
               if pathlib.Path(p).exists()), None)
if not chrome:
    sys.exit("Chrome not found")
pdf = here / "QSCHED.pdf"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", html.as_uri()], check=True, timeout=120)
html.unlink()
print("wrote", pdf)
