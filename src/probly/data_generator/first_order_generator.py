"""Extended and Improved Prototype implementation of First-Order data generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import json

import torch
import torch.nn.functional as F


def _is_probabilities(x: torch.Tensor, atol: float = 1e-4) -> bool:
    """(For to_probs idk might delete/change later) check if tensor looks like probabilities along last dim.

    Conditions:
    - all values in [0, 1]
    - rows sum approximately to 1 (within atol)
    """
    if x.numel() == 0:
        return False
    min_ok = torch.all(x >= -atol)
    max_ok = torch.all(x <= 1 + atol)
    if not (min_ok and max_ok):
        return False
    sums = x.sum(dim=-1)
    return torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0)


@dataclass
class FirstOrderDataGenerator:
    """Minimal First-Order data generator.

    Parameters
    ----------
    model:
        A Callable that maps a batch of inputs to logits or probs.
        Normally a `torch.nn.Module`.
    device:
        Device for inference (e.g., 'cpu' or 'cuda'). Default 'cpu'.
    batch_size:
        Batch size to use when wrapping a Dataset. (Default now down 64 instead of 128.)
    output_mode:
        One of {'auto', 'logits', 'probs'}. If 'auto', attempt to detect whether
        outputs are logits or probabilities. If 'logits', apply softmax. If 'probs',
        use as is. Default of course 'auto'.
    output_transform:
        func to convert raw model output to probs. If called
        this is over `output_mode`.
    input_getter:
        func to extract model input from dataset item.
        Signature: input_getter(sample) -> model_input
        When None expects dataset items to be (input, target) or input only.
    model_name:
        Optional string identifier. (saved with metadata)
    """

    model: Callable[..., Any]
    device: str = "cpu"
    batch_size: int = 64
    output_mode: str = "auto"  # your options: 'auto' | 'logits' | 'probs'
    output_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    input_getter: Callable[[Any], Any] | None = None
    model_name: str | None = None

    def to_device(self, x: Any) -> Any:
        """Move tensor/nested tensors to the same device if applicable."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, (list, tuple)):
            return type(x)(self.to_device(xx) for xx in x)
        if isinstance(x, Mapping):
            return type(x)({k: self.to_device(v) for k, v in x.items()})
        return x

    def to_probs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to probabilities."""
        if self.output_transform is not None:
            return self.output_transform(outputs)

        mode = self.output_mode.lower()
        if mode == "probs":
            return outputs
        if mode == "logits":
            return F.softmax(outputs, dim=-1)
        # auto
        return outputs if _is_probabilities(outputs) else F.softmax(outputs, dim=-1)

    def prepares_batch_inp(self, sample: Any) -> Any:
        """Prepare model input from dataset sample."""
        if self.input_getter is not None:
            return self.input_getter(sample)
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            return sample[0]
        return sample

    def extract_input(self, sample: Any) -> Any:
        """Extract model input from dataset sample."""
        if self.input_getter is not None:
            return self.input_getter(sample)
        # Default conventions: either (input, target) or input only
        if isinstance(sample, (list, tuple)) and len(sample) >= 1:
            return sample[0]
        return sample

    @torch.no_grad()
    def generate_distributions(
        self,
        dataset_or_loader: Any,
        *,
        progress: bool = True,
    ) -> dict[int, list[float]]:
        """Generate per-sample probability distributions.

        Parameters
        ----------
        dataset_or_loader:
            A `torch.utils.data.Dataset` or `torch.utils.data.DataLoader`.
            Items should be tensors or tuples/dicts that have tensors.
        progress:
            If True prints simple progress information in terminal output for user to see that progress is happening.

        Returns
        -------
        dict[int, list[float]]
            Mapping from dataset index to list of probabilities.
        """
        # Remember Blatt3: Prepare the loader
        if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
            loader = dataset_or_loader
            dataset_len = len(loader.dataset) if loader.dataset is not None else None
        else:
            dataset = dataset_or_loader
            dataset_len = len(dataset)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model = self.model.to(self.device) if hasattr(self.model, "to") else self.model
        if hasattr(self.model, "eval"):
            self.model.eval()

        distributions: dict[int, list[float]] = {}
        start_idx = 0
        # print in batch-loop: show progress
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            inpt = self.prepares_batch_inp(batch)
            inpt = self.to_device(inpt)
            outputs = self.model(inpt)
            if not isinstance(outputs, torch.Tensor):
                msg = "Model must return a torch.Tensor (logits or probs)."
                raise TypeError(msg)
            probs = self.to_probs(outputs)
            if probs.ndim == 1:
                probs = probs.unsqueeze(0)

            probs_np = probs.detach().cpu().numpy()

            batch_size = probs_np.shape[0]
            for i in range(batch_size):
                idx = start_idx + i
                distributions[idx] = probs_np[i].tolist()

            start_idx += batch_size
            if progress:
                # showing minimal textual progress
                print(f"[FirstOrderDataGenerator] Batch {batch_idx + 1}/{total_batches}\r", end="")

        # newline after progress
        if progress:
            print()

        # warn if generated count differs from dataset length
        if dataset_len is not None and len(distributions) != dataset_len:
            # Do not raise hard error (streaming loaders may mismatch) just warn
            print(
                f"[FirstOrderDataGenerator] WARNING ('>_<): generated {len(distributions)} distributions,"
                f" but dataset length is {dataset_len}."
            )

        return distributions

# JSON save/load methods
    def save_distributions(
        self,
        path: str | Path,
        distributions: Mapping[int, Iterable[float]],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Save distributions and minimal metadata as JSON. Non readable  -> readable (later during load: see comment in method load_distributions)
        """
        path = Path(path)
        serializable = {
            "meta": {
                "model_name": self.model_name,
                **(meta or {}),
            },
            "distributions": {str(k): list(v) for k, v in distributions.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)

    def load_distributions(self, path: str | Path) -> tuple[dict[int, list[float]], dict[str, Any]]:
        """Load distributions and metadata from JSON.

        Returns
        -------
        (distributions, meta)
            distributions: dict[int, list[float]]
            meta: dict with any metadata saved alongside distributions
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {}) or {}
        dists_raw = obj.get("distributions", {}) or {}
        # Convert keys back to int
        distributions: dict[int, list[float]] = {int(k): list(v) for k, v in dists_raw.items()}
        return distributions, meta
