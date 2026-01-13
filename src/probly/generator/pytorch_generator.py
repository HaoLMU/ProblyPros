from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from .base_generator import BaseGenerator


class PyTorchGenerator(BaseGenerator):
    """
    PyTorch 版本：使用 torch.save / torch.load 保存与加载 tensor_dict。
    """

    @staticmethod
    def _summarize_tensor_dict(tensor_dict: Dict[str, Any]) -> str:
        # 生成一个“Tensor 体量报告”shape\dtype\内存大小
        lines: list[str] = []
        total_mb = 0.0
        for key, val in tensor_dict.items():
            if not isinstance(val, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for key='{key}', got {type(val)}")
            size_mb = val.element_size() * val.nelement() / (1024**2)
            total_mb += size_mb
            lines.append(f"  - {key}: shape={tuple(val.shape)}, dtype={val.dtype}, {size_mb:.2f} MB, device={val.device}")
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
        将 Tensor 字典保存为 .pt/.pth 文件
        """
        # 强制 tensor_dict 是 Dict[str, Any] 形式
        self._validate_mapping(tensor_dict)
        # 确保文件路径有合法的后缀
        save_path = self._ensure_suffix(save_path, (".pt", ".pth"), ".pt")
        # 在需要时自动创建保存目录
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
        从 .pt/.pth 文件加载 Tensor 字典

        device:
          - None：保持原设备（torch.load 默认行为）
          - 'cpu'/'cuda:0'：使用 map_location 映射
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File not found: {load_path}")

        tensor_dict = torch.load(load_path, map_location=device)

        if not isinstance(tensor_dict, dict):
            raise TypeError(f"Loaded object is not a dict, got {type(tensor_dict)}")

        # 强校验：确保仍是 Tensor dict
        for k, v in tensor_dict.items():
            if not isinstance(k, str):
                raise TypeError(f"Loaded dict key is not str: key={k} type={type(k)}")
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Loaded dict value is not torch.Tensor: key='{k}', type={type(v)}")

        if verbose:
            print(f"Tensor dict loaded from: {load_path}")
            print("Content summary:")
            print(self._summarize_tensor_dict(tensor_dict))

        return tensor_dict
