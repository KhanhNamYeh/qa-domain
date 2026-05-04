"""
Precompute frozen-encoder features for the entire dataset and save them as numpy
arrays. Subsequent training runs read these arrays directly — both SigLIP2 and
PhoBERT-v2 are skipped at train time.

Saved layout per split:

    <out_dir>/<split>/
        img_feats.npy   (N, 196, image_hidden_dim)   float16
        txt_feats.npy   (N, max_q_len, text_hidden_dim) float16
        txt_masks.npy   (N, max_q_len)               int64
        ans_ins.npy     (N, max_a_len)               int64  (PAD-padded)
        ans_outs.npy    (N, max_a_len)               int64  (PAD-padded)
        raw.json        list of {"question", "answer"}
        meta.json       hyperparams used for the dump
"""

import os
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from ..models.encoders import _find_encoder_layers
from .vi_segment import segment as vi_segment


def _load_image_backbone(model_name: str, device: str):
    full = AutoModel.from_pretrained(model_name)
    vision = full.vision_model if hasattr(full, "vision_model") else full
    vision.eval().to(device)
    for p in vision.parameters():
        p.requires_grad = False
    return vision


def _load_text_backbone(model_name: str, device: str):
    bert = AutoModel.from_pretrained(model_name)
    bert.eval().to(device)
    for p in bert.parameters():
        p.requires_grad = False
    return bert


@torch.no_grad()
def precompute_split(
    json_path: str,
    tokenizer,
    image_processor,
    siglip_path: str,
    phobert_path: str,
    image_root: str,
    out_dir: str,
    *,
    max_question_len: int = 32,
    max_answer_len: int = 64,
    image_layer_idx: int = -2,
    text_last_n_layers: int = 4,
    num_visual_tokens: int = 196,
    image_hidden_dim: int = 768,
    text_hidden_dim: int = 768,
    batch_size: int = 32,
    device: str = "cuda",
    save_dtype: str = "float16",
) -> None:
    """Run the two frozen backbones over every sample in `json_path` and dump
    everything needed for cached training/eval into `out_dir`."""

    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    N = len(samples)

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    np_dtype = np.float16 if save_dtype == "float16" else np.float32
    img_feats = np.zeros((N, num_visual_tokens, image_hidden_dim), dtype=np_dtype)
    txt_feats = np.zeros((N, max_question_len, text_hidden_dim), dtype=np_dtype)
    txt_masks = np.zeros((N, max_question_len), dtype=np.int64)
    ans_ins = np.full((N, max_answer_len), pad_id, dtype=np.int64)
    ans_outs = np.full((N, max_answer_len), pad_id, dtype=np.int64)
    raw_meta = []

    vision = _load_image_backbone(siglip_path, device)
    bert = _load_text_backbone(phobert_path, device)

    # Hooks
    captured_img = [None]

    def img_hook(module, inputs, output):
        captured_img[0] = output[0] if isinstance(output, tuple) else output

    h1 = _find_encoder_layers(vision)[image_layer_idx].register_forward_hook(img_hook)

    captured_txt = [None] * text_last_n_layers
    handles = []
    for slot, layer in enumerate(_find_encoder_layers(bert)[-text_last_n_layers:]):
        def make_hook(i):
            def hook(module, inputs, output):
                captured_txt[i] = output[0] if isinstance(output, tuple) else output
            return hook
        handles.append(layer.register_forward_hook(make_hook(slot)))

    try:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = samples[start:end]
            B = len(batch)

            # ---- Image branch ----
            pvs = []
            for s in batch:
                img = Image.open(os.path.join(image_root, s["image_path"])).convert("RGB")
                pv = image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
                pvs.append(pv)
            pixel_values = torch.stack(pvs).to(device)

            captured_img[0] = None
            _ = vision(pixel_values=pixel_values)
            h_img = captured_img[0]
            if h_img is None:
                raise RuntimeError("Image hook failed to capture output.")
            if h_img.shape[1] == 197:
                h_img = h_img[:, 1:, :]
            img_feats[start:end] = h_img.to(getattr(torch, save_dtype)).cpu().numpy()

            # ---- Question branch ----
            # PhoBERT-v2 expects word-segmented input (compound words joined with _).
            enc = tokenizer(
                [vi_segment(s["question"]) for s in batch],
                truncation=True,
                max_length=max_question_len,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)

            for slot in range(text_last_n_layers):
                captured_txt[slot] = None
            _ = bert(input_ids=input_ids, attention_mask=attn_mask)
            if any(c is None for c in captured_txt):
                raise RuntimeError("Text hooks did not all fire.")
            stacked = torch.stack(captured_txt, dim=0)        # (n, B, T, C)
            mean_feat = stacked.mean(dim=0)                   # (B, T, C)

            T = mean_feat.shape[1]
            if T < max_question_len:
                pad = torch.zeros(B, max_question_len - T, text_hidden_dim, device=device)
                mean_feat = torch.cat([mean_feat, pad], dim=1)
                attn_mask = torch.nn.functional.pad(attn_mask, (0, max_question_len - T), value=0)
            else:
                mean_feat = mean_feat[:, :max_question_len, :]
                attn_mask = attn_mask[:, :max_question_len]

            txt_feats[start:end] = mean_feat.to(getattr(torch, save_dtype)).cpu().numpy()
            txt_masks[start:end] = attn_mask.cpu().numpy()

            # ---- Answer tokenization (no encoder needed) ----
            # Segment first so the decoder learns to emit PhoBERT-style tokens;
            # we'll undo the underscore at metric time.
            for i, s in enumerate(batch):
                ids = tokenizer(
                    vi_segment(s["answer"]),
                    truncation=True,
                    max_length=max_answer_len - 2,
                    add_special_tokens=False,
                )["input_ids"]
                ans_in_seq = [bos_id] + list(ids)
                ans_out_seq = list(ids) + [eos_id]
                Lin = min(len(ans_in_seq), max_answer_len)
                Lout = min(len(ans_out_seq), max_answer_len)
                ans_ins[start + i, :Lin] = ans_in_seq[:Lin]
                ans_outs[start + i, :Lout] = ans_out_seq[:Lout]
                raw_meta.append({"question": s["question"], "answer": s["answer"]})

            print(f"  [{end}/{N}]", end="\r")
    finally:
        h1.remove()
        for h in handles:
            h.remove()

    print()
    np.save(os.path.join(out_dir, "img_feats.npy"), img_feats)
    np.save(os.path.join(out_dir, "txt_feats.npy"), txt_feats)
    np.save(os.path.join(out_dir, "txt_masks.npy"), txt_masks)
    np.save(os.path.join(out_dir, "ans_ins.npy"), ans_ins)
    np.save(os.path.join(out_dir, "ans_outs.npy"), ans_outs)
    with open(os.path.join(out_dir, "raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw_meta, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_samples": N,
            "max_question_len": max_question_len,
            "max_answer_len": max_answer_len,
            "image_hidden_dim": image_hidden_dim,
            "text_hidden_dim": text_hidden_dim,
            "num_visual_tokens": num_visual_tokens,
            "image_layer_idx": image_layer_idx,
            "text_last_n_layers": text_last_n_layers,
            "save_dtype": save_dtype,
            "pad_id": pad_id, "bos_id": bos_id, "eos_id": eos_id,
        }, f, ensure_ascii=False, indent=2)
    print(f"  saved -> {out_dir}")


def is_cached(out_dir: str) -> bool:
    must = ["img_feats.npy", "txt_feats.npy", "txt_masks.npy",
            "ans_ins.npy", "ans_outs.npy", "raw.json", "meta.json"]
    return all(os.path.exists(os.path.join(out_dir, f)) for f in must)
