from typing import Any, Dict, Optional
from pathlib import Path

import torch


def save_distributions(
    distributions: Dict[str, Any],
    save_path: str | Path,
    create_dir: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Save distributions to a torch binary file (.pt / .pth).
    """
    path = Path(save_path)

    if path.suffix == "":
        path = path.with_suffix(".pt")
    elif path.suffix not in {".pt", ".pth"}:
        raise ValueError("File suffix must be '.pt' or '.pth'.")

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(distributions, path)

    if verbose:
        print(f"Saved distributions to: {path}")

    return path


def load_distributions(
    load_path: str | Path,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Load distributions from a torch binary file (.pt / .pth).
    """
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    distributions = torch.load(path, map_location=device)

    if not isinstance(distributions, dict):
        raise TypeError("Loaded object is not a dictionary.")

    if verbose:
        print(f"Loaded distributions from: {path}")

    return distributions
