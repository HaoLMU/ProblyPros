"""Prototype for FirstOrderGenerator (p(y|x)) data."""

from typing import Any
import json
import torch
import torch.nn.functional as F


class FirstOrderDataGenerator:
    def __init__(self, model: Any, batch_size: int = 128) -> None:
        self.model = model
        self.batch_size = batch_size

    def prepares_batch_inp(self, batch: Any) -> Any:
        if isinstance(batch, (list, tuple)):
            if len(batch) > 0 and isinstance(batch[0], (str, bytes)):
                return batch  # pass list of texts to HF pipeline
            return batch[0]
        return batch

    def to_probs(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return F.softmax(x, dim=-1)

    @torch.no_grad()
    def generate_distributions(self, dataset_or_loader: Any) -> dict[int, list[float]]:
        if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
            loader = dataset_or_loader
        else:
            loader = torch.utils.data.DataLoader(dataset_or_loader, batch_size=self.batch_size, shuffle=False)
        if hasattr(self.model, "eval"):
            self.model.eval()
        out: dict[int, list[float]] = {}
        i = 0
        for batch in loader:
            x = self.prepares_bacth_inp(batch)
            y = self.model(x)
            # two naive branches: torch tensor logits OR HF pipeline outputs
            if isinstance(y, torch.Tensor):
                p = self.to_probs(y).detach().cpu().tolist()
            elif isinstance(y, list):
                # expect: list of list of {"label": str, "score": float}
                p = []
                for yy in y:
                    if isinstance(yy, list) and yy and isinstance(yy[0], dict) and "score" in yy[0]:
                        p.append([float(d.get("score", 0.0)) for d in yy])
                    else:
                        raise TypeError("weird pipeline output format ?")
            else:
                raise TypeError("model must return either torch.Tensor or HF pipeline output (see code comments)")
            for row in p:
                out[i] = row
                i += 1
        return out

    def save_distributions(self, path: str, distributions: dict[int, list[float]]) -> None:
        data = {"distributions": {str(k): v for k, v in distributions.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_distributions(self, path: str) -> dict[int, list[float]]:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        d = obj.get("distributions", {})
        return {int(k): list(v) for k, v in d.items()}


def make_hf_text_pipeline(model_name: str, device: int = -1):
    from transformers import pipeline
    return pipeline("text-classification", model=model_name, device=device, return_all_scores=True)
