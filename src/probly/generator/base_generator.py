from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Mapping


@dataclass(frozen=True)
class SaveLoadConfig:
    create_dir: bool = False
    verbose: bool = True


class BaseGenerator(ABC):
    """
    Base class for generators.

    This class only defines the interface for saving and loading
    distribution data.

    Constraints:
    - Subclasses must implement save_distributions and load_distributions.
    - tensor_dict must be a Mapping[str, Any], and the accepted value types
      (e.g. torch / tensorflow tensors) are defined by the subclass.
    """

    @staticmethod
    def _ensure_suffix(
        path: str,
        allowed_suffixes: tuple[str, ...],
        default_suffix: str,
    ) -> str:
        # Ensure that the file path has a valid suffix
        if not path.endswith(allowed_suffixes):
            return f"{path}{default_suffix}"
        return path

    @staticmethod
    def _maybe_create_dir(save_path: str, create_dir: bool) -> None:
        # Create the target directory if required
        if not create_dir:
            return
        dir_name = os.path.dirname(save_path)
        # save_path may be something like "a.pt", where dirname == ""
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def _validate_mapping(tensor_dict: Mapping[str, Any]) -> None:
        # Enforce tensor_dict to be of type Mapping[str, Any]
        if not isinstance(tensor_dict, Mapping):
            raise TypeError(
                f"tensor_dict must be a Mapping[str, Any], got {type(tensor_dict)}"
            )
        for k in tensor_dict.keys():
            if not isinstance(k, str):
                raise TypeError(
                    f"tensor_dict keys must be str, got key={k} type={type(k)}"
                )

    @abstractmethod
    def save_distributions(
        self,
        tensor_dict: Dict[str, Any],
        save_path: str,
        create_dir: bool = False,
        verbose: bool = True,
    ) -> None:
        """Save distribution data (serialization is framework-specific)."""
        raise NotImplementedError

    @abstractmethod
    def load_distributions(
        self,
        load_path: str,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Load distribution data (deserialization is framework-specific)."""
        raise NotImplementedError
