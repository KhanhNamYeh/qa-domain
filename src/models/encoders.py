import torch
import torch.nn as nn
from transformers import AutoModel


def _find_encoder_layers(module: nn.Module):
    """
    Walk a HF model and return the first ModuleList that looks like a stack
    of transformer encoder layers. Handles SigLIP / CLIP / BERT variants.
    """
    candidates = []
    for name, sub in module.named_modules():
        if isinstance(sub, nn.ModuleList) and len(sub) >= 4:
            # Heuristic: layers usually live under '.encoder.layers' or '.encoder.layer'
            if name.endswith(("encoder.layers", "encoder.layer", "layers", "layer")):
                candidates.append((name, sub))
    if not candidates:
        raise RuntimeError(f"Could not locate encoder layers in {type(module).__name__}")
    # Prefer the deepest match (most specific)
    candidates.sort(key=lambda x: -len(x[0]))
    return candidates[0][1]


class ImageEncoder(nn.Module):
    """
    Wraps SigLIP2-B/16. Frozen. Returns 196 patch tokens captured by a forward
    hook on the encoder layer at index `layer_idx` (default -2 = penultimate).
    Output shape: (B, 196, out_dim)

    Hook-based extraction avoids relying on `output_hidden_states`, which some
    transformers versions ignore for SigLIP and return None.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        out_dim: int = 512,
        layer_idx: int = -2,
    ):
        super().__init__()
        full = AutoModel.from_pretrained(model_name)
        self.vision = full.vision_model if hasattr(full, "vision_model") else full
        self.layer_idx = layer_idx

        in_dim = getattr(self.vision.config, "hidden_size", 768)
        self.proj = nn.Linear(in_dim, out_dim)

        # Register a forward hook on the chosen encoder layer.
        layers = _find_encoder_layers(self.vision)
        target = layers[layer_idx]
        self._captured = None

        def _hook(module, inputs, output):
            # Layer outputs are usually a tuple (hidden, ...). Take the tensor.
            self._captured = output[0] if isinstance(output, tuple) else output

        self._hook_handle = target.register_forward_hook(_hook)

        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.eval()
        return self

    @torch.no_grad()
    def _backbone_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._captured = None
        _ = self.vision(pixel_values=pixel_values)
        h = self._captured
        if h is None:
            raise RuntimeError("Hook on SigLIP encoder layer did not capture any output.")
        # SigLIP vision uses no CLS token; if a variant adds one, strip it.
        if h.shape[1] == 197:
            h = h[:, 1:, :]
        return h  # (B, 196, C)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self._backbone_forward(pixel_values)
        return self.proj(h)


class QuestionEncoder(nn.Module):
    """
    Wraps PhoBERT-v2. Frozen. Returns token-level features computed as the
    mean of the last `last_n_layers` encoder-layer outputs (captured via
    forward hooks). Output shape: (B, T_q, out_dim) and the original mask.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        out_dim: int = 512,
        last_n_layers: int = 4,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.last_n_layers = last_n_layers

        in_dim = getattr(self.bert.config, "hidden_size", 768)
        self.proj = nn.Linear(in_dim, out_dim)

        layers = _find_encoder_layers(self.bert)
        self._buffers_list = [None] * last_n_layers
        self._handles = []
        for slot, layer in enumerate(layers[-last_n_layers:]):
            def make_hook(i):
                def _hook(module, inputs, output):
                    self._buffers_list[i] = output[0] if isinstance(output, tuple) else output
                return _hook
            self._handles.append(layer.register_forward_hook(make_hook(slot)))

        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    @torch.no_grad()
    def _backbone_forward(self, input_ids, attention_mask):
        self._buffers_list = [None] * self.last_n_layers
        _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if any(b is None for b in self._buffers_list):
            raise RuntimeError("Hooks on PhoBERT encoder layers did not all fire.")
        stacked = torch.stack(self._buffers_list, dim=0)  # (n, B, T, C)
        return stacked.mean(dim=0)                        # (B, T, C)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self._backbone_forward(input_ids, attention_mask)
        return self.proj(h), attention_mask
