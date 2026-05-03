"""
Entry point for training a single configuration.

Usage:
    python -m src.train --config A1 \\
        --train_json qa_data/train.json --val_json qa_data/val.json \\
        --test_json qa_data/test.json --image_root .
"""

import argparse
import random

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .build import (
    build_tokenizer_and_processor,
    resolve_special_ids,
    build_loaders,
    build_model,
)
from .training import Trainer, Evaluator
from .metrics import ExactMatch, BLEUScore, METEORScore


CONFIGS = {
    "A1": dict(decoder_type="lstm",        norm_type="layernorm", ffn_type="vanilla", run_name="A1_lstm"),
    "A2": dict(decoder_type="transformer", norm_type="layernorm", ffn_type="vanilla", run_name="A2_transformer_vanilla"),
    "A3": dict(decoder_type="transformer", norm_type="rmsnorm",   ffn_type="swiglu",  run_name="A3_transformer_modern"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_json",   required=True)
    p.add_argument("--test_json",  required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    cfg_overrides = CONFIGS[args.config]
    model_cfg = ModelConfig(
        decoder_type=cfg_overrides["decoder_type"],
        norm_type=cfg_overrides["norm_type"],
        ffn_type=cfg_overrides["ffn_type"],
    )
    train_cfg = TrainConfig(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        image_root=args.image_root,
        run_name=cfg_overrides["run_name"],
    )
    if args.epochs is not None:     train_cfg.epochs = args.epochs
    if args.batch_size is not None: train_cfg.batch_size = args.batch_size
    if args.lr is not None:         train_cfg.lr = args.lr

    set_seed(train_cfg.seed)

    tokenizer, image_processor = build_tokenizer_and_processor(model_cfg)
    model_cfg = resolve_special_ids(tokenizer, model_cfg)

    train_loader, val_loader, _ = build_loaders(model_cfg, train_cfg, tokenizer, image_processor)
    model = build_model(model_cfg)

    metrics = [ExactMatch(), BLEUScore(), METEORScore()]
    evaluator = Evaluator(
        tokenizer=tokenizer,
        metrics=metrics,
        bos_id=model_cfg.bos_id,
        eos_id=model_cfg.eos_id,
        pad_id=model_cfg.pad_id,
        max_len=model_cfg.max_answer_len,
        beam_size=train_cfg.beam_size,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        evaluator=evaluator,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
