"""
Factory helpers: build tokenizer, image processor, dataset/loader, and the full
VQAModel for a given (ModelConfig, TrainConfig) pair. Used by train.py / eval.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor

from .config import ModelConfig, TrainConfig
from .data import VQADataset, VQACollator
from .models import (
    ImageEncoder,
    QuestionEncoder,
    CrossAttentionFusion,
    LSTMDecoder,
    TransformerDecoder,
    VQAModel,
    RMSNorm,
    SwiGLU,
    VanillaFFN,
)


def build_tokenizer_and_processor(model_cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.text_encoder_name, use_fast=False)
    image_processor = AutoImageProcessor.from_pretrained(model_cfg.image_encoder_name)
    return tokenizer, image_processor


def resolve_special_ids(tokenizer, model_cfg: ModelConfig) -> ModelConfig:
    model_cfg.vocab_size = tokenizer.vocab_size if model_cfg.vocab_size is None else model_cfg.vocab_size
    model_cfg.pad_id = tokenizer.pad_token_id
    model_cfg.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    model_cfg.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    return model_cfg


def build_loaders(model_cfg: ModelConfig, train_cfg: TrainConfig, tokenizer, image_processor):
    pad_id = tokenizer.pad_token_id
    collator = VQACollator(pad_id=pad_id)

    train_ds = VQADataset(
        train_cfg.train_json, tokenizer, image_processor,
        image_root=train_cfg.image_root,
        max_question_len=train_cfg.max_question_len,
        max_answer_len=train_cfg.max_answer_len,
    )
    val_ds = VQADataset(
        train_cfg.val_json, tokenizer, image_processor,
        image_root=train_cfg.image_root,
        max_question_len=train_cfg.max_question_len,
        max_answer_len=train_cfg.max_answer_len,
    )
    test_ds = VQADataset(
        train_cfg.test_json, tokenizer, image_processor,
        image_root=train_cfg.image_root,
        max_question_len=train_cfg.max_question_len,
        max_answer_len=train_cfg.max_answer_len,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, collate_fn=collator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, collate_fn=collator, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, collate_fn=collator, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _build_decoder(model_cfg: ModelConfig) -> nn.Module:
    if model_cfg.decoder_type == "lstm":
        return LSTMDecoder(
            vocab_size=model_cfg.vocab_size,
            pad_id=model_cfg.pad_id,
            embed_dim=model_cfg.hidden_dim,
            hidden_dim=model_cfg.hidden_dim,
            memory_dim=model_cfg.hidden_dim,
            dropout=model_cfg.dropout,
        )

    norm_cls = RMSNorm if model_cfg.norm_type == "rmsnorm" else nn.LayerNorm
    ffn_cls = SwiGLU if model_cfg.ffn_type == "swiglu" else VanillaFFN
    return TransformerDecoder(
        vocab_size=model_cfg.vocab_size,
        pad_id=model_cfg.pad_id,
        dim=model_cfg.hidden_dim,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_decoder_layers,
        ffn_dim=model_cfg.ffn_dim,
        dropout=model_cfg.dropout,
        max_len=model_cfg.max_answer_len,
        norm_cls=norm_cls,
        ffn_cls=ffn_cls,
    )


def build_model(model_cfg: ModelConfig) -> VQAModel:
    img_enc = ImageEncoder(
        model_name=model_cfg.image_encoder_name,
        out_dim=model_cfg.hidden_dim,
        layer_idx=model_cfg.image_layer_idx,
    )
    txt_enc = QuestionEncoder(
        model_name=model_cfg.text_encoder_name,
        out_dim=model_cfg.hidden_dim,
        last_n_layers=model_cfg.text_last_n_layers,
    )
    fusion = CrossAttentionFusion(
        dim=model_cfg.hidden_dim,
        n_heads=model_cfg.fusion_n_heads,
        dropout=model_cfg.fusion_dropout,
    )
    decoder = _build_decoder(model_cfg)
    return VQAModel(img_enc, txt_enc, fusion, decoder)
