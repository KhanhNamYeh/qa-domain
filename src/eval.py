"""
Standalone evaluation on the test split, including BERTScore (which is heavy
so it's only run here, not during validation).

Usage:
    python -m src.eval --config A2 --ckpt checkpoints/A2_transformer_vanilla_final.pt \\
        --train_json qa_data/train.json --val_json qa_data/val.json \\
        --test_json qa_data/test.json --image_root .
"""

import argparse

import torch

from .config import ModelConfig, TrainConfig
from .build import (
    build_tokenizer_and_processor,
    resolve_special_ids,
    build_loaders,
    build_model,
)
from .training import Evaluator
from .metrics import ExactMatch, BLEUScore, METEORScore, BERTScoreMetric


CONFIGS = {
    "A1": dict(decoder_type="lstm",        norm_type="layernorm", ffn_type="vanilla"),
    "A2": dict(decoder_type="transformer", norm_type="layernorm", ffn_type="vanilla"),
    "A3": dict(decoder_type="transformer", norm_type="rmsnorm",   ffn_type="swiglu"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_json",   required=True)
    p.add_argument("--test_json",  required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--beam_size", type=int, default=1)
    args = p.parse_args()

    over = CONFIGS[args.config]
    model_cfg = ModelConfig(
        decoder_type=over["decoder_type"],
        norm_type=over["norm_type"],
        ffn_type=over["ffn_type"],
    )
    train_cfg = TrainConfig(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        image_root=args.image_root,
        beam_size=args.beam_size,
    )

    tokenizer, image_processor = build_tokenizer_and_processor(model_cfg)
    model_cfg = resolve_special_ids(tokenizer, model_cfg)

    _, _, test_loader = build_loaders(model_cfg, train_cfg, tokenizer, image_processor)
    model = build_model(model_cfg)

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    metrics = [ExactMatch(), BLEUScore(), METEORScore(), BERTScoreMetric()]
    evaluator = Evaluator(
        tokenizer=tokenizer,
        metrics=metrics,
        bos_id=model_cfg.bos_id,
        eos_id=model_cfg.eos_id,
        pad_id=model_cfg.pad_id,
        max_len=model_cfg.max_answer_len,
        beam_size=args.beam_size,
    )

    results = evaluator.evaluate(model, test_loader)
    print(f"Results for {args.config} (ckpt={args.ckpt}):")
    for k, v in results.items():
        print(f"  {k:12s} = {v:.4f}")


if __name__ == "__main__":
    main()
