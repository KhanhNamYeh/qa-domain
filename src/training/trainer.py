import os
import time
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Standard train loop:
      - Adam(W) optimizer
      - CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
      - Linear Teacher Forcing Ratio decay (used by decoders that opt in;
        the decoders here use full Teacher Forcing for simplicity, the schedule
        is logged for transparency and can be wired to scheduled sampling later)
      - Periodic checkpointing + simple text logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        train_cfg,
        model_cfg,
        evaluator=None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.evaluator = evaluator

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=model_cfg.pad_id,
            label_smoothing=train_cfg.label_smoothing,
        )

        os.makedirs(train_cfg.ckpt_dir, exist_ok=True)
        os.makedirs(train_cfg.log_dir, exist_ok=True)
        self.log_path = os.path.join(train_cfg.log_dir, f"{train_cfg.run_name}.log")

    # ------------------------------------------------------------------
    def _tfr(self, epoch: int) -> float:
        cfg = self.train_cfg
        if cfg.epochs <= 1:
            return cfg.tfr_end
        frac = epoch / (cfg.epochs - 1)
        return cfg.tfr_start + (cfg.tfr_end - cfg.tfr_start) * frac

    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _save_ckpt(self, tag: str) -> None:
        path = os.path.join(self.train_cfg.ckpt_dir, f"{self.train_cfg.run_name}_{tag}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_cfg": asdict(self.model_cfg),
                "train_cfg": asdict(self.train_cfg),
            },
            path,
        )
        self._log(f"Saved checkpoint -> {path}")

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            q_ids = batch["question_ids"].to(self.device, non_blocking=True)
            q_mask = batch["question_mask"].to(self.device, non_blocking=True)
            ans_in = batch["answer_in"].to(self.device, non_blocking=True)
            ans_out = batch["answer_out"].to(self.device, non_blocking=True)

            logits = self.model(pixel_values, q_ids, q_mask, ans_in)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), ans_out.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.train_cfg.grad_clip,
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self):
        cfg = self.train_cfg
        for epoch in range(cfg.epochs):
            tfr = self._tfr(epoch)
            t0 = time.time()
            train_loss = self.train_one_epoch(epoch)
            dt = time.time() - t0
            self._log(
                f"epoch={epoch+1}/{cfg.epochs} train_loss={train_loss:.4f} "
                f"tfr={tfr:.3f} time={dt:.1f}s"
            )

            if self.evaluator is not None and self.val_loader is not None and (epoch + 1) % cfg.eval_every == 0:
                results = self.evaluator.evaluate(self.model, self.val_loader)
                metric_str = " ".join(f"{k}={v:.4f}" for k, v in results.items())
                self._log(f"epoch={epoch+1} VAL {metric_str}")

            if (epoch + 1) % cfg.save_every == 0:
                self._save_ckpt(f"epoch{epoch+1}")

        self._save_ckpt("final")
