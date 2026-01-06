import re
from typing import Iterable


_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPE_MAP.get(ch, ch) for ch in s)


def format_value_for_latex(v: str) -> str:
    """
    Escapes text; preserves plain numbers.
    """
    vv = v.strip()
    # If it looks like a float/int (optionally scientific notation), don't escape.
    if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", vv):
        return vv
    return latex_escape(vv)


def to_latex_table(
    row_order: list[str],
    columns: dict[str, dict[str, str]],
    caption: str = "Latest results per experiment type",
    label: str = "tab:latest-results",
    column_align: str | None = None,
    use_booktabs: bool = True,
) -> str:
    exp_types = list(columns.keys())
    n_cols = 1 + len(exp_types)

    # Alignment string, e.g. "l" + "c"*k
    if column_align is None:
        column_align = "l" + "c" * (n_cols - 1)

    # Header row
    header_cells = ["Metric", *exp_types]
    header = " & ".join(latex_escape(h) for h in header_cells) + r" \\"

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(rf"\begin{{tabular}}{{{column_align}}}")

    if use_booktabs:
        lines.append(r"\toprule")
        lines.append(header)
        lines.append(r"\midrule")
    else:
        lines.append(r"\hline")
        lines.append(header)
        lines.append(r"\hline")

    for metric in row_order:
        row_cells = [metric] + [columns[e].get(metric, "") for e in exp_types]
        row = " & ".join(format_value_for_latex(c) for c in row_cells) + r" \\"
        lines.append(row)

    if use_booktabs:
        lines.append(r"\bottomrule")
    else:
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"
