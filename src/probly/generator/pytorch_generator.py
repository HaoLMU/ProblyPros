from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from .base_generator import BaseGenerator


class PyTorchGenerator(BaseGenerator):
    """
    PyTorch implementation using torch.save and torch.load
    to serialize and deserialize tensor dictionaries.
    """

    @staticmethod
    def _summarize_tensor_dict(tensor_dict: Dict[str, Any]) -> str:
        # Generate a summary of tensor shapes, dtypes, devices, and memory usage
        lines: list[str] = []
        total_mb = 0.0
        for key, val in tensor_dict.items():
            if not isinstance(val, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor for key='{key}', got {type(val)}"
                )
            size_mb = val.element_size() * val.nelement() / (1024**2)
            total_mb += size_mb
            lines.append(
                f"  - {key}: shape={tuple(val.shape)}, dtype={val.dtype}, "
                f"{size_mb:.2f} MB, device={val.device}"
            )
        lines.append(f"Total size: {total_mb:.2f} MB")
        return "\n".join(lines)

    def save_distributions(
        self,
        tensor_dict: Dict[str, Any],
        save_path: str,
        create_dir: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Save a dictionary of tensors to a .pt or .pth file.
        """
        # Ensure tensor_dict follows the expected Mapping[str, Any] format
        self._validate_mapping(tensor_dict)
        # Ensure the file path has a valid suffix
        save_path = self._ensure_suffix(save_path, (".pt", ".pth"), ".pt")
        # Create the target directory if required
        self._maybe_create_dir(save_path, create_dir)

        torch.save(tensor_dict, save_path)

        if verbose:
            print(f"Tensor dict saved to: {save_path}")
            print("Content summary:")
            print(self._summarize_tensor_dict(tensor_dict))

    def load_distributions(
        self,
        load_path: str,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a dictionary of tensors from a .pt or .pth file.

        device:
          - None: keep the original device (default torch.load behavior)
          - 'cpu' / 'cuda:0': use map_location to remap tensors
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File not found: {load_path}")

        tensor_dict = torch.load(load_path, map_location=device)

        if not isinstance(tensor_dict, dict):
            raise TypeError(
                f"Loaded object is not a dict, got {type(tensor_dict)}"
            )

        # Strict validation: ensure the loaded object is still a tensor dict
        for k, v in tensor_dict.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"Loaded dict key is not str: key={k} type={type(k)}"
                )
            if not isinstance(v, torch.Tensor):
                raise TypeError(
                    f"Loaded dict value is not torch.Tensor: key='{k}', type={type(v)}"
                )

        if verbose:
            print(f"Tensor dict loaded from: {load_path}")
            print("Content summary:")
            print(self._summarize_tensor_dict(tensor_dict))

        return tensor_dict
