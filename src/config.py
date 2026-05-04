from dataclasses import dataclass, asdict
from typing import Literal, Optional
import json
import os


@dataclass
class ModelConfig:
    # Encoders
    image_encoder_name: str = "google/siglip2-base-patch16-224"
    text_encoder_name: str = "vinai/phobert-base-v2"
    image_hidden_dim: int = 768
    text_hidden_dim: int = 768
    image_layer_idx: int = -2          # penultimate SigLIP layer
    text_last_n_layers: int = 4        # mean of last 4 PhoBERT layers
    num_visual_tokens: int = 196       # 14x14 patches

    # Common projection dim
    hidden_dim: int = 512

    # Fusion
    fusion_n_heads: int = 8
    fusion_dropout: float = 0.1
    n_fusion_layers: int = 3

    # Decoder
    decoder_type: Literal["lstm", "transformer"] = "transformer"
    n_decoder_layers: int = 3       # transformer decoder block count
    n_lstm_layers: int = 3          # LSTM stacked-cell count
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    max_answer_len: int = 64

    # Ablation switches (only used when decoder_type == "transformer")
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    ffn_type: Literal["vanilla", "swiglu"] = "vanilla"

    # Vocab (resolved at runtime from PhoBERT tokenizer)
    vocab_size: Optional[int] = None
    pad_id: Optional[int] = None
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None


@dataclass
class TrainConfig:
    # ---- REQUIRED: data paths ----
    # Đặt ở đầu để bắt buộc người dùng truyền — không có default đường dẫn
    # tương đối kiểu "qa_data/train.json" để tránh phụ thuộc CWD.
    train_json: str
    val_json: str
    test_json: str
    image_root: str

    # ---- Optional: optimization ----
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    epochs: int = 30
    grad_clip: float = 1.0
    label_smoothing: float = 0.1

    # Teacher forcing ratio decay (linear)
    tfr_start: float = 1.0
    tfr_end: float = 0.5

    # Data sizing / loaders
    max_question_len: int = 32
    max_answer_len: int = 64
    num_workers: int = 4

    # Logging / checkpoint
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 1
    eval_every: int = 1
    seed: int = 42

    # Inference
    beam_size: int = 1                 # 1 == greedy

    # Run id (used to namespace ckpt/logs)
    run_name: str = "A2_transformer_vanilla"


def save_config(cfg, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def load_config(cls, path: str):
    with open(path, "r", encoding="utf-8") as f:
        return cls(**json.load(f))
