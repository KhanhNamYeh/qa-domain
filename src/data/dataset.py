import json
import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class VQADataset(Dataset):
    """
    Reads JSON annotations of the form:
        [{ "image_path": "images/xxx.jpg",
           "question":   "...",
           "answer":     "..." }, ...]

    Tokenizes question and answer with PhoBERT tokenizer, prepares Teacher Forcing
    pair (ans_in, ans_out) where:
        ans_in  = [BOS, t1, t2, ..., tN]
        ans_out = [t1, t2, ..., tN, EOS]
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        image_processor,
        image_root: str = ".",
        max_question_len: int = 32,
        max_answer_len: int = 64,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = image_root
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len

        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    def __len__(self):
        return len(self.samples)

    def _load_image(self, rel_path: str):
        path = os.path.join(self.image_root, rel_path)
        img = Image.open(path).convert("RGB")
        out = self.image_processor(images=img, return_tensors="pt")
        # SigLIP returns "pixel_values" with shape (1, 3, H, W)
        return out["pixel_values"].squeeze(0)

    def _encode_question(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_question_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def _encode_answer(self, text: str):
        # Tokenize without adding the model's default special tokens, then
        # build BOS/EOS bracketed sequences manually.
        ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_answer_len - 2,
            add_special_tokens=False,
        )["input_ids"]
        ids = torch.tensor(ids, dtype=torch.long)
        bos = torch.tensor([self.bos_id], dtype=torch.long)
        eos = torch.tensor([self.eos_id], dtype=torch.long)
        ans_in = torch.cat([bos, ids], dim=0)
        ans_out = torch.cat([ids, eos], dim=0)
        return ans_in, ans_out

    def __getitem__(self, idx):
        s = self.samples[idx]
        pixel_values = self._load_image(s["image_path"])
        q_ids, q_mask = self._encode_question(s["question"])
        ans_in, ans_out = self._encode_answer(s["answer"])
        return {
            "pixel_values": pixel_values,
            "question_ids": q_ids,
            "question_mask": q_mask,
            "answer_in": ans_in,
            "answer_out": ans_out,
            "raw_question": s["question"],
            "raw_answer": s["answer"],
        }
