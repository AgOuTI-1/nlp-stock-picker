from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS = ["negative", "neutral", "positive"]


@dataclass
class FinBertScorer:
    model_name: str = "ProsusAI/finbert"
    device: str | None = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def score_batch(self, texts: List[str], max_length: int = 64) -> np.ndarray:
        if not texts:
            return np.array([])
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        # score = P(pos) - P(neg)
        pos = probs[:, LABELS.index("positive")]
        neg = probs[:, LABELS.index("negative")]
        return pos - neg
