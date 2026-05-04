from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader


class Evaluator:
    """
    Runs autoregressive generation on a DataLoader and aggregates metrics.
    `metrics` is a list of BaseMetric instances; they are reset before use.
    """

    def __init__(
        self,
        tokenizer,
        metrics: List,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_len: int = 64,
        beam_size: int = 1,
        device: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_len = max_len
        self.beam_size = beam_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _decode_ids(self, id_seq: torch.Tensor) -> str:
        ids = id_seq.tolist()
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i == self.pad_id or i == self.bos_id:
                continue
            out.append(i)
        text = self.tokenizer.decode(out, skip_special_tokens=True).strip()
        # The decoder emits PhoBERT-segmented tokens (compound words joined with
        # `_`). The references in `raw_answers` are unsegmented; convert back
        # so BLEU/EM/METEOR compare like-for-like.
        return text.replace("_", " ")

    @torch.no_grad()
    def evaluate(self, model, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        for m in self.metrics:
            m.reset()

        for batch in loader:
            tensors = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            tensors.pop("answer_in", None)
            tensors.pop("answer_out", None)
            pred_ids = model.generate(
                **tensors,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                max_len=self.max_len,
                beam_size=self.beam_size,
            )
            preds = [self._decode_ids(seq) for seq in pred_ids]
            targets = batch["raw_answers"]

            for m in self.metrics:
                m.update(preds, targets)

        return {m.name: m.compute() for m in self.metrics}
