import torch
import torch.nn as nn
from transformers import AutoModel


class ImageEncoder(nn.Module):
    """
    Wraps SigLIP2-B/16. Frozen. Returns 196 patch tokens taken from the
    penultimate hidden layer (-2), then projected to `out_dim`.
    Output shape: (B, 196, out_dim)
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        out_dim: int = 512,
        layer_idx: int = -2,
    ):
        super().__init__()
        full = AutoModel.from_pretrained(model_name)
        # We only need the vision tower
        self.vision = full.vision_model if hasattr(full, "vision_model") else full
        self.layer_idx = layer_idx

        # Force the model to actually emit hidden_states; some HF versions
        # ignore the forward kwarg and only honour the config flag.
        self.vision.config.output_hidden_states = True

        in_dim = getattr(self.vision.config, "hidden_size", 768)
        self.proj = nn.Linear(in_dim, out_dim)

        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the vision backbone frozen / in eval mode regardless.
        self.vision.eval()
        return self

    @torch.no_grad()
    def _backbone_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = getattr(out, "hidden_states", None)
        if hs is None and isinstance(out, dict):
            hs = out.get("hidden_states")
        if hs is None:
            raise RuntimeError(
                "SigLIP vision_model did not return hidden_states; "
                "check transformers version (>=4.45 expected)."
            )
        h = hs[self.layer_idx]
        # SigLIP vision uses no CLS token; if some variant adds one strip it.
        if h.shape[1] == 197:
            h = h[:, 1:, :]
        return h  # (B, 196, C)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self._backbone_forward(pixel_values)
        return self.proj(h)


class QuestionEncoder(nn.Module):
    """
    Wraps PhoBERT-v2. Frozen. Returns token-level features computed as the
    mean of the last `last_n_layers` hidden states, projected to `out_dim`.
    Output shape: (B, T_q, out_dim) and the original attention mask.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        out_dim: int = 512,
        last_n_layers: int = 4,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.config.output_hidden_states = True
        self.last_n_layers = last_n_layers

        in_dim = getattr(self.bert.config, "hidden_size", 768)
        self.proj = nn.Linear(in_dim, out_dim)

        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    @torch.no_grad()
    def _backbone_forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = getattr(out, "hidden_states", None)
        if hs is None and isinstance(out, dict):
            hs = out.get("hidden_states")
        if hs is None:
            raise RuntimeError(
                "PhoBERT did not return hidden_states; check transformers version."
            )
        stacked = torch.stack(list(hs[-self.last_n_layers:]), dim=0)  # (n, B, T, C)
        return stacked.mean(dim=0)                                    # (B, T, C)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self._backbone_forward(input_ids, attention_mask)
        return self.proj(h), attention_mask
