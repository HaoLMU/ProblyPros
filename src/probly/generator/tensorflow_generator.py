from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from .base_generator import BaseGenerator


class TensorFlowGenerator(BaseGenerator):
    """
    TensorFlow implementation.

    - Uses np.savez_compressed to store tensors as a .npz file
      (keys are preserved).
    - Loaded arrays are converted back to tf.Tensor.
    """

    @staticmethod
    def _summarize_tensor_dict(tensor_dict: Dict[str, Any]) -> str:
        # Generate a summary of tensor shapes, dtypes, and memory usage
        lines: list[str] = []
        total_mb = 0.0
        for key, val in tensor_dict.items():
            if not isinstance(val, (tf.Tensor, tf.Variable)):
                raise TypeError(
                    f"Expected tf.Tensor or tf.Variable for key='{key}', got {type(val)}"
                )
            t = tf.convert_to_tensor(val)
            nbytes = int(tf.size(t).numpy()) * t.dtype.size
            size_mb = nbytes / (1024**2)
            total_mb += size_mb
            lines.append(
                f"  - {key}: shape={tuple(t.shape)}, dtype={t.dtype.name}, "
                f"{size_mb:.2f} MB"
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
        TensorFlow does not provide an equivalent of torch.save(dict_of_tensors).

        This method stores a TensorFlow tensor dictionary as a compressed
        .npz file.
        """
        self._validate_mapping(tensor_dict)

        save_path = self._ensure_suffix(save_path, (".npz",), ".npz")
        self._maybe_create_dir(save_path, create_dir)

        arrays: Dict[str, np.ndarray] = {}
        for k, v in tensor_dict.items():
            if not isinstance(v, (tf.Tensor, tf.Variable)):
                raise TypeError(
                    f"Expected tf.Tensor or tf.Variable for key='{k}', got {type(v)}"
                )
            arrays[k] = tf.convert_to_tensor(v).numpy()

        np.savez_compressed(save_path, **arrays)

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
        Load a tensor dictionary from a .npz file.

        device:
          - None: use the default device
          - '/CPU:0', '/GPU:0', etc.: create tensors on the specified device
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File not found: {load_path}")

        loaded = np.load(load_path, allow_pickle=False)

        tensor_dict: Dict[str, Any] = {}
        if device:
            with tf.device(device):
                for k in loaded.files:
                    tensor_dict[k] = tf.convert_to_tensor(loaded[k])
        else:
            for k in loaded.files:
                tensor_dict[k] = tf.convert_to_tensor(loaded[k])

        if verbose:
            print(f"Tensor dict loaded from: {load_path}")
            print("Content summary:")
            print(self._summarize_tensor_dict(tensor_dict))

        return tensor_dict
