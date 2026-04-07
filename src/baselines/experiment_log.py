from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "docs" / "baselines"
LOG_PATH = LOG_DIR / "EXPERIMENT_LOG.md"


def _format_value(value: Any) -> str:
    """Format one value for the markdown experiment log."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def reset_run_log() -> None:
    """Create a fresh experiment log with a stable header."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Experiment Log",
        "",
        "This log records the execution order, parameters, results, and outputs",
        "for the baseline experiments under `src/baselines/`.",
        "",
        "- Scope: `src/baselines/`",
        "- Outputs: `outputs/baselines/`",
        "- Notes: each run appends its rationale, main parameters, key metrics, and artifacts.",
        "",
    ]
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def append_run_log(
    *,
    run_name: str,
    rationale: str,
    script_path: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    outputs: list[str],
) -> None:
    """Append one experiment entry to the baseline experiment log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        reset_run_log()

    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.extend(
        [
            f"## {timestamp} | {run_name}",
            "",
            f"- Script: `{script_path}`",
            f"- Rationale: {rationale}",
            "- Params:",
        ]
    )
    for key, value in params.items():
        lines.append(f"  - `{key}`: `{_format_value(value)}`")

    lines.append("- Metrics:")
    for key, value in metrics.items():
        lines.append(f"  - `{key}`: `{_format_value(value)}`")

    lines.append("- Outputs:")
    for output in outputs:
        lines.append(f"  - `{output}`")

    lines.append("")

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
