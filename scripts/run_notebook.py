"""Execute a notebook in-process and save its outputs, without needing a Jupyter kernel (ipykernel).

Supports what the project notebooks use: printed output, the value of a cell's last expression
(including DataFrame HTML), `display(...)`, and Matplotlib figures. It does not support IPython
magics or shell commands. On Colab or a normal Jupyter setup, just run the notebook as usual.

Usage (from the project root):  python scripts/run_notebook.py notebooks/01_eda.ipynb [...]
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _rich(obj) -> dict:
    data = {"text/plain": repr(obj)}
    html = getattr(obj, "_repr_html_", None)
    if callable(html):
        rendered = html()
        if rendered:
            data["text/html"] = rendered
    markdown = getattr(obj, "_repr_markdown_", None)  # e.g. IPython.display.Markdown
    if callable(markdown):
        rendered = markdown()
        if rendered:
            data["text/markdown"] = rendered
    png = getattr(obj, "_repr_png_", None)  # e.g. IPython.display.Image
    if callable(png):
        raw = png()
        if raw:
            data["image/png"] = base64.b64encode(raw).decode() if isinstance(raw, bytes) else raw
    return data


def execute(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    previous_cwd = Path.cwd()
    os.chdir(path.parent)  # same working directory Jupyter would use
    namespace: dict = {"__name__": "__main__"}
    outputs: list = []
    stream = io.StringIO()

    def flush_stream() -> None:
        text = stream.getvalue()
        if text:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
            stream.seek(0)
            stream.truncate()

    def capture_figures(*_args, **_kwargs) -> None:
        flush_stream()
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(buf.getvalue()).decode(),
                                                                      "text/plain": "<Figure>"}))
        plt.close("all")

    def display(*objects, **_kwargs) -> None:
        flush_stream()
        for obj in objects:
            outputs.append(nbformat.v4.new_output("display_data", data=_rich(obj)))

    plt.show = capture_figures
    namespace["display"] = display
    count = 0
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            outputs.clear()
            tree = ast.parse(cell.source)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            result = None
            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    exec(compile(tree, f"<cell {count}>", "exec"), namespace)
                    if last is not None:
                        result = eval(compile(ast.Expression(last.value), f"<cell {count}>", "eval"), namespace)
            except Exception as exc:
                flush_stream()
                outputs.append(nbformat.v4.new_output("error", ename=type(exc).__name__, evalue=str(exc),
                                                      traceback=traceback.format_exc().splitlines()))
                cell.outputs, cell.execution_count = list(outputs), count
                nbformat.write(nb, path)
                raise RuntimeError(f"{path.name}: cell {count} failed") from exc
            capture_figures()
            if result is not None:
                outputs.append(nbformat.v4.new_output("execute_result", data=_rich(result), execution_count=count))
            cell.outputs, cell.execution_count = list(outputs), count
    finally:
        os.chdir(previous_cwd)
    nbformat.write(nb, path)
    print(f"executed {path} ({count} code cells)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        execute(Path(arg).resolve())
