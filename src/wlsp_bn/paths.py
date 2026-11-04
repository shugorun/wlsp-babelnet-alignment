"""Where the data of the pipeline is.

The external resources are found through configs/paths.toml, which each user
writes by copying configs/paths.example.toml. The file is not committed,
because the paths differ between machines and the resources themselves
cannot be redistributed (see docs/data_sources.md).

Everything the scripts build from those resources goes to data/interim/,
which is not committed either, since it holds WLSP and BabelNet content.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "paths.toml"
EXAMPLE = ROOT / "configs" / "paths.example.toml"

DATA = ROOT / "data"
INTERIM = DATA / "interim"
ANNOTATIONS = DATA / "annotations"
TERM_EXPANSION = DATA / "term_expansion"
SPLITS = DATA / "splits"
PREDICTIONS = DATA / "predictions"
PROMPTS = ROOT / "prompts"


def resource(key: str) -> Path:
    """Return the path given for `key` in configs/paths.toml.

    A relative path is taken relative to the repository root.

    Raises
    ------
    FileNotFoundError
        If configs/paths.toml does not exist.
    KeyError
        If the file does not give `key`.
    """
    if not CONFIG.exists():
        raise FileNotFoundError(
            f"{CONFIG} does not exist; copy {EXAMPLE.name} to paths.toml "
            "and write the locations of the resources on this machine"
        )
    with CONFIG.open("rb") as f:
        paths = tomllib.load(f)["paths"]
    if key not in paths:
        raise KeyError(f"{CONFIG} gives no path for {key!r} under [paths]")
    path = Path(paths[key])
    return path if path.is_absolute() else ROOT / path
