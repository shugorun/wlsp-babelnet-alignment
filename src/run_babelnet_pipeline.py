# run_babelnet_pipeline.py
# Convert pkl files to parquet, run the BabelNet pipeline in .venv38,
# then convert the updated parquet files back to pkl.

from pathlib import Path
import os
import socket
import subprocess
import time

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

# Files to convert before/after the pipeline.
# Add more files here if needed.
FILES_TO_CONVERT = [
    ROOT / "data" / "gold" / "gold_B_records.pkl",
    ROOT / "data" / "processed" / "babelnet_.pkl",
]

# Python 3.8 environment used for BabelNet.
VENV_DIR = ROOT / ".venv38"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_BABELNET_RPC = VENV_DIR / "Scripts" / "babelnet-rpc.exe"

# BabelNet RPC settings.
BABELNET_DIR = Path(os.environ.get("BABELNET_DIR", "Path-to-BabelNet"))
RPC_HOST = "127.0.0.1"
RPC_PORT = 7790

# Scripts to run in order.
PIPELINE_SCRIPTS = [
    ROOT / "src" / "add_sids_to_wlsp.py",
    ROOT / "src" / "add_babelnet_rows.py",
]


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


def start_rpc_server() -> subprocess.Popen:
    """Start babelnet-rpc in the Python 3.8 virtual environment."""
    if str(BABELNET_DIR) == "Path-to-BabelNet":
        raise ValueError("Set BABELNET_DIR before running this script.")

    cmd = [
        str(VENV_BABELNET_RPC),
        "start",
        "--bn",
        str(BABELNET_DIR),
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


def run_script(script_path: Path) -> None:
    """Run one pipeline script in the Python 3.8 virtual environment."""
    cmd = [str(VENV_PYTHON), str(script_path)]
    print(f"Running: {script_path}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    # Convert input files from pkl to parquet before using the BabelNet environment.
    parquet_files = []
    for pkl_path in FILES_TO_CONVERT:
        if not pkl_path.exists():
            raise FileNotFoundError(pkl_path)
        parquet_files.append(pkl_to_parquet(pkl_path))

    rpc_proc = None
    try:
        # Start RPC and run the pipeline scripts.
        rpc_proc = start_rpc_server()
        for script_path in PIPELINE_SCRIPTS:
            run_script(script_path)

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
