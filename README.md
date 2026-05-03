# VQA Múa Lân — Vietnamese Visual Question Answering

Dự án triển khai bài toán Visual Question Answering tiếng Việt trên dataset *Múa Lân* theo kiến trúc **rời (encoder-decoder)** đã được đơn giản hoá hoàn toàn — không bbox, không spatial attention dựa trên bbox.

- **Image encoder (frozen):** SigLIP2-B/16, lấy đặc trưng từ layer áp chót (−2) → 196 patch token.
- **Question encoder (frozen):** PhoBERT-v2, mean của 4 layer cuối.
- **Fusion:** Cross-attention (text Q × visual KV) sau khi cùng chiếu về 512 chiều.
- **Decoder:** Hoán đổi được giữa LSTM (A1), Transformer cổ điển (A2), Transformer modern stack với RMSNorm + SwiGLU (A3).
- **Loss:** `CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)` — thuần, không Focal/Weighted CE.

---

## Cấu trúc thư mục

```
DL/
├── images/                            # Toàn bộ ảnh — không thay đổi
├── qa_data/                           # Annotation JSON (đã loại bbox)
│   ├── train.json                     # 9879 cặp Q/A
│   ├── val.json                       # 1235 cặp
│   └── test.json                      # 1235 cặp
│
├── src/                               # Toàn bộ code OOP
│   ├── config.py                      # @dataclass ModelConfig, TrainConfig
│   ├── build.py                       # Factory: tokenizer, loaders, model
│   ├── train.py                       # Entry CLI: python -m src.train --config A1|A2|A3
│   ├── eval.py                        # Entry CLI: chạy trên test split + BERTScore
│   │
│   ├── data/
│   │   ├── dataset.py                 # VQADataset (đọc qa_data/*.json, tokenize Q/A)
│   │   └── collator.py                # VQACollator (pad batch, attn mask)
│   │
│   ├── models/
│   │   ├── encoders.py                # ImageEncoder (SigLIP2 frozen, layer −2)
│   │   │                              # QuestionEncoder (PhoBERT-v2 frozen, mean 4 last)
│   │   ├── fusion.py                  # CrossAttentionFusion (text × visual)
│   │   ├── blocks.py                  # RMSNorm, SwiGLU, VanillaFFN
│   │   ├── decoders.py                # BaseDecoder (abstract)
│   │   │                              # LSTMDecoder            ← A1
│   │   │                              # TransformerDecoderBlock
│   │   │                              # TransformerDecoder     ← A2 / A3
│   │   └── vqa_model.py               # VQAModel (composition + DI cho decoder)
│   │
│   ├── metrics/
│   │   ├── base.py                    # BaseMetric (interface)
│   │   ├── exact_match.py             # ExactMatch (VQA Accuracy)
│   │   ├── bleu.py                    # BLEUScore (NLTK / sacrebleu fallback)
│   │   ├── meteor.py                  # METEORScore (NLTK)
│   │   └── bertscore.py               # BERTScoreMetric (multilingual)
│   │
│   └── training/
│       ├── trainer.py                 # Trainer (AdamW, CE+label_smoothing, TFR log, ckpt)
│       └── evaluator.py               # Evaluator (greedy/beam generate + metrics)
│
├── main.ipynb                         # Train 2 cấu hình tối ưu: A1 (LSTM) + A3 (modern)
│                                      # — vẽ line chart train_loss, val_acc, val_bleu, val_meteor
│
├── checkpoints/                       # Tự sinh khi train (.pt theo run_name)
├── logs/                              # Log text + history JSON cho biểu đồ
└── requirements.txt
```

---

## Cấu hình thực nghiệm

| Tag | Decoder       | Norm      | FFN        | Vai trò |
|-----|---------------|-----------|------------|---------|
| A1  | LSTM          | —         | —          | Baseline rời |
| A2  | Transformer   | LayerNorm | VanillaFFN | Transformer cổ điển (sẵn sàng nhưng không train trong notebook chính) |
| A3  | Transformer   | RMSNorm   | SwiGLU     | Modern stack (LLaMA/Gemma/Qwen) |

`main.ipynb` huấn luyện **A1** và **A3** — hai cấu hình tối ưu nhất ở hai cực thiết kế (RNN cổ điển vs Transformer modern). A2 vẫn được hỗ trợ qua CLI (`python -m src.train --config A2`).

---

## Sử dụng

### 1. Cài đặt

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Notebook (cách dùng chính)

Mở `main.ipynb` → chạy hết cell. Train xong sẽ:
- Lưu checkpoint vào `checkpoints/A1_lstm_final.pt` và `checkpoints/A3_transformer_modern_final.pt`.
- Lưu lịch sử train/val vào `logs/main_history.json`.
- Vẽ biểu đồ line: `train_loss`, `val_exact_match`, `val_bleu`, `val_meteor`.

### 3. CLI

```bash
# Train một cấu hình
python -m src.train --config A1
python -m src.train --config A2
python -m src.train --config A3

# Eval trên test split (kèm BERTScore)
python -m src.eval --config A3 --ckpt checkpoints/A3_transformer_modern_final.pt
```

---

## Điểm thiết kế OOP cốt lõi

- **Polymorphism qua `BaseDecoder`** — `VQAModel` chỉ phụ thuộc vào interface chung. Đổi A1 ↔ A2 ↔ A3 chỉ là đổi class truyền vào constructor.
- **Constructor injection cho `TransformerDecoderBlock`** — một class duy nhất nhận `norm_cls` và `ffn_cls` làm tham số → A2 vs A3 chỉ là đổi 2 dòng config, không duplicate code.
- **Tách biệt config khỏi logic** — toàn bộ hyperparameter nằm trong `@dataclass`, dễ serialize ra JSON cho reproducibility.
- **Metrics đồng nhất qua `BaseMetric`** — `Evaluator` gọi `update()` / `compute()` thống nhất cho mọi loại metric.
