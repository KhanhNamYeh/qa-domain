import torch
from torch.nn.utils.rnn import pad_sequence


class VQACollator:
    """
    Pads a batch produced by VQADataset:
      - pixel_values stacked into (B, 3, H, W)
      - question_ids / question_mask padded to max question length in batch
      - answer_in / answer_out padded to max answer length in batch (PAD = pad_id, ignored in loss)
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

        q_ids = pad_sequence(
            [b["question_ids"] for b in batch],
            batch_first=True,
            padding_value=self.pad_id,
        )
        q_mask = pad_sequence(
            [b["question_mask"] for b in batch],
            batch_first=True,
            padding_value=0,
        )

        ans_in = pad_sequence(
            [b["answer_in"] for b in batch],
            batch_first=True,
            padding_value=self.pad_id,
        )
        ans_out = pad_sequence(
            [b["answer_out"] for b in batch],
            batch_first=True,
            padding_value=self.pad_id,
        )

        return {
            "pixel_values": pixel_values,
            "question_ids": q_ids,
            "question_mask": q_mask,
            "answer_in": ans_in,
            "answer_out": ans_out,
            "raw_questions": [b["raw_question"] for b in batch],
            "raw_answers": [b["raw_answer"] for b in batch],
        }
