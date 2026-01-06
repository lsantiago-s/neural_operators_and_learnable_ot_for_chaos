from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from helpers_for_latex import to_latex_table


TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class RunInfo:
    exp_type: str
    exp_hash: str
    run_dir: Path
    timestamp: datetime
    timestamp_raw: str


def parse_timestamp(ts: str) -> datetime:
    # Example: "20251118_105630"
    return datetime.strptime(ts, TIMESTAMP_FMT)


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None
    return data


def parse_results_txt(path: Path) -> dict[str, str]:
    """
    Parses lines of the form "Key: Value" into an ordered dict (in file order).
    Skips empty lines. Leaves values as strings (you can cast later if desired).
    """
    results: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return results

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            # If a malformed line appears, keep it but avoid crashing.
            results[line] = ""
            continue
        key, value = line.split(":", 1)
        results[key.strip()] = value.strip()
    return results


def find_latest_run_per_exp_type(outputs_root: Path) -> dict[str, RunInfo]:
    """
    Assumes structure: outputs_root/exp_type/exp_hash/{config.json, results.txt}
    Traverses exp_type directories and selects the exp_hash with the most recent timestamp.
    """
    latest: dict[str, RunInfo] = {}

    if not outputs_root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {outputs_root}")

    for exp_type_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        exp_type = exp_type_dir.name

        for exp_hash_dir in sorted(p for p in exp_type_dir.iterdir() if p.is_dir()):
            config_path = exp_hash_dir / "config.json"
            results_path = exp_hash_dir / "results.txt"
            if not config_path.is_file() or not results_path.is_file():
                continue

            cfg = safe_read_json(config_path)
            if cfg is None or "timestamp" not in cfg:
                continue

            ts_raw = cfg["timestamp"]
            if not isinstance(ts_raw, str):
                continue

            try:
                ts = parse_timestamp(ts_raw)
            except ValueError:
                # Timestamp doesn't match expected format
                continue

            candidate = RunInfo(
                exp_type=exp_type,
                exp_hash=exp_hash_dir.name,
                run_dir=exp_hash_dir,
                timestamp=ts,
                timestamp_raw=ts_raw,
            )

            current = latest.get(exp_type)
            if current is None or candidate.timestamp > current.timestamp:
                latest[exp_type] = candidate

    return latest


def build_table(latest_runs: dict[str, RunInfo]) -> tuple[list[str], dict[str, dict[str, str]]]:
    """
    Returns (row_order, columns_dict) where:
      - row_order is a stable list of keys (metrics) in a preferred order
      - columns_dict maps exp_type -> {metric_key -> value_str}
    """
    # Parse all results first
    columns: dict[str, dict[str, str]] = {}
    for exp_type, run in sorted(latest_runs.items()):
        results = parse_results_txt(run.run_dir / "results.txt")
        # Add a couple useful metadata fields for traceability
        results.setdefault("Exp hash", run.exp_hash)
        results.setdefault("Timestamp", run.timestamp_raw)
        columns[exp_type] = results

    # Preferred row ordering (based on your example); then add any extras at the end.
    preferred = [
        "Summary type",
        "Projected state",
        "OT",
        "Noise",
        "Histogram Error",
        "Energy Spec. Error",
        "IPM",
        "Timestamp",
        "Exp hash",
    ]

    all_keys: set[str] = set()
    for col in columns.values():
        all_keys.update(col.keys())

    row_order: list[str] = [k for k in preferred if k in all_keys]
    row_order.extend(sorted(all_keys - set(row_order)))

    return row_order, columns


def to_csv(row_order: list[str], columns: dict[str, dict[str, str]]) -> str:
    # Minimal CSV writer to avoid pandas dependency.
    exp_types = list(columns.keys())
    header = ["Metric", *exp_types]

    def esc(s: str) -> str:
        if any(ch in s for ch in [",", '"', "\n"]):
            return '"' + s.replace('"', '""') + '"'
        return s

    lines: list[str] = [",".join(esc(h) for h in header)]
    for metric in row_order:
        row = [metric]
        for exp_type in exp_types:
            row.append(columns[exp_type].get(metric, ""))
        lines.append(",".join(esc(x) for x in row))
    return "\n".join(lines) + "\n"


def to_markdown(row_order: list[str], columns: dict[str, dict[str, str]]) -> str:
    exp_types = list(columns.keys())
    header = ["Metric", *exp_types]
    sep = ["---"] * len(header)

    def cell(s: str) -> str:
        return s.replace("|", "\\|")

    lines: list[str] = [
        "| " + " | ".join(cell(h) for h in header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for metric in row_order:
        row = [metric] + [columns[e].get(metric, "") for e in exp_types]
        lines.append("| " + " | ".join(cell(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect latest results.txt per exp_type (by config.json timestamp) and build a combined table."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root outputs folder containing exp_type/exp_hash/ subfolders (default: outputs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder for aggregated files. Defaults to --root if omitted.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the markdown table to stdout.",
    )
    args = parser.parse_args()

    outputs_root: Path = args.root
    out_dir: Path = args.out if args.out is not None else outputs_root
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = find_latest_run_per_exp_type(outputs_root)
    if not latest:
        raise RuntimeError(
            f"No valid runs found under {outputs_root}. "
            "Expected outputs/exp_type/exp_hash/{config.json,results.txt} with config.json['timestamp']."
        )

    row_order, columns = build_table(latest)

    csv_text = to_csv(row_order, columns)
    md_text = to_markdown(row_order, columns)
    tex_text = to_latex_table(
        row_order,
        columns,
        caption="Latest results per experiment type",
        label="tab:latest-results",
        use_booktabs=True,
    )

    csv_path = out_dir / "latest_results.csv"
    md_path = out_dir / "latest_results.md"
    tex_path = out_dir / "latest_results.tex"

    csv_path.write_text(csv_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    tex_path.write_text(tex_text, encoding="utf-8")

    if args.print:
        print(md_text)

    # Small summary to stderr-friendly stdout
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print("Latest runs picked:")
    for exp_type, run in sorted(latest.items()):
        print(f"  - {exp_type}: {run.exp_hash} ({run.timestamp_raw})")


if __name__ == "__main__":
    # On some systems, os.walk order varies; we already sort iterators for stability.
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
