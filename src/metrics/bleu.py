from typing import List

from .base import BaseMetric


class BLEUScore(BaseMetric):
    """Corpus-level BLEU-4 with smoothing. Uses NLTK if available, else sacrebleu."""

    name = "bleu"

    def __init__(self, n: int = 4):
        self.n = n
        self.refs: List[List[str]] = []
        self.hyps: List[str] = []

    def update(self, preds: List[str], targets: List[str]) -> None:
        for p, t in zip(preds, targets):
            self.hyps.append(p.strip())
            self.refs.append([t.strip()])

    def compute(self) -> float:
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            sm = SmoothingFunction().method1
            refs_tok = [[r.split() for r in rs] for rs in self.refs]
            hyps_tok = [h.split() for h in self.hyps]
            weights = tuple([1.0 / self.n] * self.n)
            return float(corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=sm))
        except Exception:
            import sacrebleu
            refs_T = [[rs[0] for rs in self.refs]]
            return float(sacrebleu.corpus_bleu(self.hyps, refs_T).score) / 100.0
