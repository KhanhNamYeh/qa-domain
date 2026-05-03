from typing import List

from .base import BaseMetric


class BERTScoreMetric(BaseMetric):
    """
    BERTScore F1, computed in one shot at .compute() time. Uses a multilingual
    model by default so Vietnamese answers are scored sensibly.
    """

    name = "bertscore"

    def __init__(self, model_type: str = "bert-base-multilingual-cased", lang: str = "vi"):
        self.model_type = model_type
        self.lang = lang
        self.preds: List[str] = []
        self.targets: List[str] = []

    def update(self, preds: List[str], targets: List[str]) -> None:
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self) -> float:
        if not self.preds:
            return 0.0
        from bert_score import score as bertscore
        _, _, f1 = bertscore(
            cands=self.preds,
            refs=self.targets,
            model_type=self.model_type,
            lang=self.lang,
            verbose=False,
        )
        return float(f1.mean().item())
