from typing import List

from .base import BaseMetric


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


class ExactMatch(BaseMetric):
    """VQA-style accuracy: 1 if normalized prediction equals normalized target, else 0."""

    name = "exact_match"

    def __init__(self):
        self.n = 0
        self.correct = 0

    def update(self, preds: List[str], targets: List[str]) -> None:
        for p, t in zip(preds, targets):
            if _normalize(p) == _normalize(t):
                self.correct += 1
            self.n += 1

    def compute(self) -> float:
        return self.correct / self.n if self.n > 0 else 0.0
