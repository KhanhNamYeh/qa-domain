"""
Memory-mapped Dataset/Collator that reads precomputed encoder features from
disk. Used in place of (VQADataset, VQACollator) when caching is enabled.
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedVQADataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.img = np.load(os.path.join(cache_dir, "img_feats.npy"), mmap_mode="r")
        self.txt = np.load(os.path.join(cache_dir, "txt_feats.npy"), mmap_mode="r")
        self.mask = np.load(os.path.join(cache_dir, "txt_masks.npy"), mmap_mode="r")
        self.ans_in = np.load(os.path.join(cache_dir, "ans_ins.npy"), mmap_mode="r")
        self.ans_out = np.load(os.path.join(cache_dir, "ans_outs.npy"), mmap_mode="r")
        with open(os.path.join(cache_dir, "raw.json"), "r", encoding="utf-8") as f:
            self.raw = json.load(f)
        with open(os.path.join(cache_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return {
            "img_feat":  torch.from_numpy(np.asarray(self.img[i],   dtype=np.float32)),
            "txt_feat":  torch.from_numpy(np.asarray(self.txt[i],   dtype=np.float32)),
            "txt_mask":  torch.from_numpy(np.asarray(self.mask[i],  dtype=np.int64)),
            "answer_in": torch.from_numpy(np.asarray(self.ans_in[i], dtype=np.int64)),
            "answer_out":torch.from_numpy(np.asarray(self.ans_out[i], dtype=np.int64)),
            "raw_question": self.raw[i]["question"],
            "raw_answer":   self.raw[i]["answer"],
        }


class CachedVQACollator:
    """Pre-padded so collator only stacks."""

    def __call__(self, batch):
        return {
            "img_feat":  torch.stack([b["img_feat"]  for b in batch], dim=0),
            "txt_feat":  torch.stack([b["txt_feat"]  for b in batch], dim=0),
            "txt_mask":  torch.stack([b["txt_mask"]  for b in batch], dim=0),
            "answer_in": torch.stack([b["answer_in"] for b in batch], dim=0),
            "answer_out":torch.stack([b["answer_out"]for b in batch], dim=0),
            "raw_questions": [b["raw_question"] for b in batch],
            "raw_answers":   [b["raw_answer"]   for b in batch],
        }
