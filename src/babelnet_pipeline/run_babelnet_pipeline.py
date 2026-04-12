#!/usr/bin/env python
# Usage: python src/babelnet_pipeline/run_babelnet_pipeline.py --records-pkl data/gold/gold_A_records.pkl --babelnet-pkl data/processed/babelnet_.pkl --term-outputs-dir outputs/api_runs/term_expansion/version_1/gold_A --babelnet-dir Path-to-BabelNet
# run_babelnet_pipeline.py
# Convert pkl files to parquet, run the BabelNet pipeline in .venv38,
# then convert the updated parquet files back to pkl.

import argparse
from pathlib import Path
import os
import socket
import subprocess
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

# Files to convert before/after the pipeline.
# Add more files here if needed.
DEFAULT_RECORDS_PKL = ROOT / "data" / "gold" / "gold_B_records.pkl"
DEFAULT_BABELNET_PKL = ROOT / "data" / "processed" / "babelnet_.pkl"
DEFAULT_TERM_OUTPUTS_DIR = ROOT / "outputs" / "api_runs" / "term_expansion" / "version_1" / "gold_A"

# Python 3.8 environment used for BabelNet.
VENV_DIR = ROOT / ".venv38"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_BABELNET_RPC = VENV_DIR / "Scripts" / "babelnet-rpc.exe"

# BabelNet RPC settings.
DEFAULT_BABELNET_DIR = Path(os.environ.get("BABELNET_DIR", "Path-to-BabelNet"))
RPC_HOST = "127.0.0.1"
RPC_PORT = 7790

# Scripts to run in order.
ADD_SIDS_SCRIPT = ROOT / "src" / "babelnet_pipeline" / "add_sids_to_wlsp.py"
ADD_BABELNET_ROWS_SCRIPT = ROOT / "src" / "babelnet_pipeline" / "add_babelnet_rows.py"
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-pkl", type=Path, default=DEFAULT_RECORDS_PKL)
    ap.add_argument("--babelnet-pkl", type=Path, default=DEFAULT_BABELNET_PKL)
    ap.add_argument("--term-outputs-dir", type=Path, default=DEFAULT_TERM_OUTPUTS_DIR)
    ap.add_argument("--babelnet-dir", type=Path, default=DEFAULT_BABELNET_DIR)
    ap.add_argument("--max-workers", type=int, default=4)
    return ap.parse_args()


def pkl_to_parquet(pkl_path: Path) -> Path:
    """Convert one pickle file to parquet."""
    parquet_path = pkl_path.with_suffix(".parquet")
    df = pd.read_pickle(pkl_path)
    df.to_parquet(parquet_path)
    print(f"Converted: {pkl_path} -> {parquet_path}")
    return parquet_path


def parquet_to_pkl(parquet_path: Path) -> Path:
    """Convert one parquet file back to pickle."""
    pkl_path = parquet_path.with_suffix(".pkl")
    df = pd.read_parquet(parquet_path)
    df.to_pickle(pkl_path)
    print(f"Converted: {parquet_path} -> {pkl_path}")
    return pkl_path


def wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    """Wait until the RPC server starts listening."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"RPC server is ready at {host}:{port}")
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"RPC server did not start within {timeout} seconds.")


def start_rpc_server(babelnet_dir: Path) -> subprocess.Popen:
    """Start babelnet-rpc in the Python 3.8 virtual environment."""
    if str(babelnet_dir) == "Path-to-BabelNet":
        raise ValueError("Set BABELNET_DIR before running this script.")

    cmd = [
        str(VENV_BABELNET_RPC),
        "start",
        "--bn",
        str(babelnet_dir),
        "--m",
        "tcp",
        "--tcp",
        str(RPC_PORT),
        "--no-doc",
    ]

    print("Starting BabelNet RPC server...")
    proc = subprocess.Popen(cmd, cwd=ROOT)
    wait_for_port(RPC_HOST, RPC_PORT)
    return proc


def stop_rpc_server(proc: subprocess.Popen) -> None:
    """Stop the RPC server process."""
    if proc.poll() is None:
        print("Stopping BabelNet RPC server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_script(script_path: Path, extra_args: list[str]) -> None:
    """Run one pipeline script in the Python 3.8 virtual environment."""
    cmd = [str(VENV_PYTHON), str(script_path), *extra_args]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()

    # Convert input files from pkl to parquet before using the BabelNet environment.
    pkl_files = [args.records_pkl, args.babelnet_pkl]
    parquet_files = []
    for pkl_path in pkl_files:
        if not pkl_path.exists():
            raise FileNotFoundError(pkl_path)
        parquet_files.append(pkl_to_parquet(pkl_path))

    records_parquet = args.records_pkl.with_suffix(".parquet")
    babelnet_parquet = args.babelnet_pkl.with_suffix(".parquet")

    rpc_proc = None
    try:
        # Start RPC and run the pipeline scripts.
        rpc_proc = start_rpc_server(args.babelnet_dir)
        run_script(
            ADD_SIDS_SCRIPT,
            [
                "--input-path",
                str(records_parquet),
                "--outputs-dir",
                str(args.term_outputs_dir),
                "--max-workers",
                str(args.max_workers),
            ],
        )
        run_script(
            ADD_BABELNET_ROWS_SCRIPT,
            [
                "--input-path",
                str(records_parquet),
                "--babelnet-path",
                str(babelnet_parquet),
                "--max-workers",
                str(args.max_workers),
            ],
        )

    finally:
        # Always stop RPC first.
        if rpc_proc is not None:
            stop_rpc_server(rpc_proc)

        # Convert the updated parquet files back to pkl.
        for parquet_path in parquet_files:
            if parquet_path.exists():
                parquet_to_pkl(parquet_path)


if __name__ == "__main__":
    main()
