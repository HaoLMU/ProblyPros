from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from .base_generator import BaseGenerator


class TensorFlowGenerator(BaseGenerator):
    """
    TensorFlow 版本：
    - 使用 np.savez_compressed 保存为 .npz（键保持不变）
    - 加载后再转回 tf.Tensor
    """

    @staticmethod
    def _summarize_tensor_dict(tensor_dict: Dict[str, Any]) -> str:
        # 生成一个“Tensor 体量报告”shape\dtype\内存大小
        lines: list[str] = []
        total_mb = 0.0
        for key, val in tensor_dict.items():
            if not isinstance(val, (tf.Tensor, tf.Variable)):
                raise TypeError(f"Expected tf.Tensor/tf.Variable for key='{key}', got {type(val)}")
            t = tf.convert_to_tensor(val)
            nbytes = int(tf.size(t).numpy()) * t.dtype.size
            size_mb = nbytes / (1024**2)
            total_mb += size_mb
            lines.append(f"  - {key}: shape={tuple(t.shape)}, dtype={t.dtype.name}, {size_mb:.2f} MB")
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
        TensorFlow 没有等价的 torch.save(dict_of_tensors)
        将 TensorFlow tensor_dict 保存为 .npz 文件（压缩）
        """
        self._validate_mapping(tensor_dict)

        save_path = self._ensure_suffix(save_path, (".npz",), ".npz")
        self._maybe_create_dir(save_path, create_dir)

        arrays: Dict[str, np.ndarray] = {}
        for k, v in tensor_dict.items():
            if not isinstance(v, (tf.Tensor, tf.Variable)):
                raise TypeError(f"Expected tf.Tensor/tf.Variable for key='{k}', got {type(v)}")
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
        从 .npz 加载为 tf.Tensor dict

        device:
          - None：默认设备
          - '/CPU:0'、'/GPU:0' 等：在指定设备上构建 tensor
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
