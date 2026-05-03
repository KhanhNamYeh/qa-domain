from typing import List

from .base import BaseMetric


class METEORScore(BaseMetric):
    """METEOR averaged over examples (NLTK implementation, tokenized by whitespace)."""

    name = "meteor"

    def __init__(self):
        self.scores: List[float] = []

    def update(self, preds: List[str], targets: List[str]) -> None:
        from nltk.translate.meteor_score import meteor_score
        for p, t in zip(preds, targets):
            self.scores.append(float(meteor_score([t.split()], p.split())))

    def compute(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0
