import os
import re
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

# Download required NLTK data for metrics
for resource in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# ---------------------------------------------------------
# 1. TEXT NORMALIZATION & UTILS
# ---------------------------------------------------------

SPECIAL_PHRASES = [
    "vàng đỏ", "đỏ sẫm", "vàng tươi", "vàng nhạt", "vàng chấm đỏ",
    "trắng cam", "trắng bạc", "trắng đen", "vàng đen", "vàng xanh",
    "đen tuyền", "trắng đỏ", "trắng hồng",
    "con lân", "người biểu diễn",
]

def normalize_special_phrases(text: str) -> str:
    """Normalize special Vietnamese phrases into single tokens."""
    normalized = str(text).lower()
    for phrase in sorted(SPECIAL_PHRASES, key=len, reverse=True):
        pattern = re.escape(phrase).replace(r"\ ", r"\s+")
        replacement = phrase.replace(" ", "_")
        normalized = re.sub(rf"\b{pattern}\b", replacement, normalized, flags=re.IGNORECASE)
    return normalized

def ids_to_words(ids, dataset):
    """Convert a list of token IDs back to a list of words, stopping at EOS."""
    words = []
    for idx in ids:
        idx = int(idx)
        if idx == dataset.eos_idx: 
            break
        if idx not in (dataset.pad_idx, dataset.sos_idx):
            words.append(dataset.idx2word.get(idx, '<UNK>'))
    return words

def _apply_rep_penalty(logits: torch.Tensor, hist: list, prev_tokens: torch.Tensor, rep_penalty: float, pad_idx: int, eos_idx: int, sos_idx: int) -> torch.Tensor:
    """Apply repetition penalty during sequence generation."""
    B, V = logits.size()
    logits = logits.clone()

    for i in range(B):
        count_map = {}
        for wi in hist[i]:
            count_map[wi] = count_map.get(wi, 0) + 1
        for wi, cnt in count_map.items():
            if 0 <= wi < V:
                pen = rep_penalty ** cnt
                logits[i, wi] = logits[i, wi] / pen if logits[i, wi] > 0 else logits[i, wi] * pen

    if prev_tokens is not None:
        for i in range(B):
            lt = int(prev_tokens[i])
            if lt not in (pad_idx, eos_idx, sos_idx) and 0 <= lt < V:
                logits[i, lt] -= 1e4

    return logits

# ---------------------------------------------------------
# 2. DATASET (For Model A1 & A2)
# ---------------------------------------------------------

MAX_BOXES = 6
BBOX_CLASS = {"con lân": 0, "người biểu diễn": 1}

class VQADataset(Dataset):
    """Dataset class for Visual Question Answering (Custom Models)."""
    def __init__(self, json_path, vocab_path, img_dir='', max_seq_len=20, transform=None, max_boxes=MAX_BOXES):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word_vocab = json.load(f)

        self.pad_idx = self.word_vocab.get('<PAD>', 0)
        self.unk_idx = self.word_vocab.get('<UNK>', 1)
        self.sos_idx = self.word_vocab.get('<SOS>', 2)
        self.eos_idx = self.word_vocab.get('<EOS>', 3)
        self.idx2word = {idx: word for word, idx in self.word_vocab.items()}

        self.img_dir = img_dir
        self.max_seq_len = max_seq_len
        self.max_boxes = max_boxes
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def tokenize_and_pad(self, text: str, add_sos=False, add_eos=False) -> torch.Tensor:
        text = normalize_special_phrases(text)
        text = re.sub(r'[^\w\s_]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        tokens = [self.word_vocab.get(w, self.unk_idx) for w in words]
        
        seq = []
        if add_sos: seq.append(self.sos_idx)
        seq.extend(tokens)
        if add_eos: seq.append(self.eos_idx)
        
        if len(seq) < self.max_seq_len:
            seq.extend([self.pad_idx] * (self.max_seq_len - len(seq)))
        else:
            seq = seq[:self.max_seq_len]
        return torch.tensor(seq, dtype=torch.long)

    def parse_bboxes(self, item: dict):
        raw_dict = item.get('bbox', {})
        boxes_flat = []
        for category, class_id in BBOX_CLASS.items():
            for box in raw_dict.get(category, []):
                if len(box) >= 4:
                    boxes_flat.append([float(class_id)] + [float(v) for v in box[:4]])
        if not boxes_flat:
            boxes_flat = [[0.0, 0.5, 0.5, 1.0, 1.0]]
        boxes_flat = boxes_flat[:self.max_boxes]
        
        arr = np.zeros((self.max_boxes, 5), dtype=np.float32)
        mask = np.zeros(self.max_boxes, dtype=np.float32)
        for i, box in enumerate(boxes_flat):
            arr[i] = box
            mask[i] = 1.0
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        image = self.transform(image)
        
        q_t = self.tokenize_and_pad(item['question'])
        ans_in = self.tokenize_and_pad(item['answer'], add_sos=True)
        ans_out = self.tokenize_and_pad(item['answer'], add_eos=True)
        bboxes, bbox_mask = self.parse_bboxes(item)
        
        return image, q_t, ans_in, ans_out, bboxes, bbox_mask

# ---------------------------------------------------------
# 3. MODEL ARCHITECTURE (Hướng A1 & A2)
# ---------------------------------------------------------

class BBoxSpatialAttention(nn.Module):
    NUM_CLASSES = 2

    def __init__(self, grid_size: int = 14, hidden_dim: int = 512):
        super().__init__()
        self.grid_size = grid_size
        self.class_embed = nn.Embedding(self.NUM_CLASSES, hidden_dim)
        self.gate_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, 1), nn.Sigmoid())
        self.bbox_geom_embed = nn.Linear(4, hidden_dim)

    def _soft_mask(self, bboxes, bbox_mask):
        B, G, dev = bboxes.size(0), self.grid_size, bboxes.device
        xs = (torch.arange(G, device=dev).float() + 0.5) / G
        ys = (torch.arange(G, device=dev).float() + 0.5) / G
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        gcx, gcy = gx.reshape(-1), gy.reshape(-1)
        cx, cy = bboxes[:, :, 1], bboxes[:, :, 2]
        w, h = bboxes[:, :, 3], bboxes[:, :, 4]
        
        x1, x2 = (cx - w / 2).clamp(0, 1), (cx + w / 2).clamp(0, 1)
        y1, y2 = (cy - h / 2).clamp(0, 1), (cy + h / 2).clamp(0, 1)
        k = 12.0
        
        def soft_in(g, lo, hi):
            return torch.sigmoid(k * (g.view(1, 1, -1) - lo.unsqueeze(2))) * torch.sigmoid(k * (hi.unsqueeze(2) - g.view(1, 1, -1)))
        
        in_box = soft_in(gcx, x1, x2) * soft_in(gcy, y1, y2)
        in_box = in_box * bbox_mask.unsqueeze(2)
        spatial = in_box.sum(dim=1)
        return spatial / spatial.amax(dim=1, keepdim=True).clamp(min=1e-6)

    def forward(self, image_features, bboxes, bbox_mask):
        spatial = self._soft_mask(bboxes, bbox_mask)
        gate = self.gate_fc(image_features)
        enhanced = image_features * (1.0 + gate * spatial.unsqueeze(2))
        valid_sum = bbox_mask.sum(1, keepdim=True).clamp(min=1)
        geom_ctx = self.bbox_geom_embed((bboxes[:, :, 1:] * bbox_mask.unsqueeze(2)).sum(1) / valid_sum).unsqueeze(1)
        cids = bboxes[:, :, 0].long().clamp(0, self.NUM_CLASSES - 1)
        cls_ctx = (self.class_embed(cids) * bbox_mask.unsqueeze(2)).sum(1, keepdim=True) / valid_sum.unsqueeze(2)
        return enhanced + 0.1 * (geom_ctx + cls_ctx)

class ImageEncoder(nn.Module):
    def __init__(self, encoded_image_size=14, hidden_dim=512, pretrained=True, trainable_cnn=False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.trainable_cnn = trainable_cnn
        if not trainable_cnn:
            for p in self.resnet.parameters(): 
                p.requires_grad = False
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.projection = nn.Linear(2048, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feature_dim = hidden_dim
        self.bbox_attn = BBoxSpatialAttention(encoded_image_size, hidden_dim)

    def forward(self, images, bboxes=None, bbox_mask=None):
        with torch.set_grad_enabled(self.trainable_cnn):
            f = self.resnet(images)
        f = self.adaptive_pool(f).permute(0, 2, 3, 1).reshape(images.size(0), -1, 2048)
        f = self.layer_norm(self.projection(f))
        if bboxes is not None and bbox_mask is not None:
            f = self.bbox_attn(f, bboxes, bbox_mask)
        return f

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, questions):
        _, (h, c) = self.lstm(self.embedding(questions))
        h_cat = torch.cat([h[0], h[1]], 1)
        c_cat = torch.cat([c[0], c[1]], 1)
        return self.layer_norm(h_cat), c_cat

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.ctx_norm = nn.LayerNorm(encoder_dim)

    def forward(self, features, h):
        att = self.full_att(torch.relu(self.enc_att(features) + self.dec_att(h).unsqueeze(1))).squeeze(2)
        alpha = torch.softmax(att, dim=1)
        return self.ctx_norm((features * alpha.unsqueeze(2)).sum(1)), alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim, attn_dim, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.attention = Attention(encoder_dim, hidden_dim, attn_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_f, q_h, q_c, ans, tfr=0.5):
        B, T = img_f.size(0), ans.size(1)
        h, c = q_h, q_c
        preds = torch.zeros(B, T, self.vocab_size, device=img_f.device)
        alphas = []
        cur = ans[:, 0]
        
        for t in range(T):
            ctx, alpha = self.attention(img_f, h)
            alphas.append(alpha.detach())
            h, c = self.lstm_cell(torch.cat([self.drop(self.embedding(cur)), ctx], 1), (h, c))
            out = self.fc(h)
            preds[:, t] = out
            cur = ans[:, t + 1] if (random.random() < tfr and t + 1 < T) else out.argmax(1)
            
        return preds, torch.stack(alphas, 1)

    def generate(self, img_f, q_h, q_c, max_len, sos_idx, eos_idx, pad_idx=0, rep_penalty=3.0, return_alphas=False):
        B = img_f.size(0)
        h, c = q_h, q_c
        cur = torch.full((B,), sos_idx, dtype=torch.long, device=img_f.device)
        done = torch.zeros(B, dtype=torch.bool, device=img_f.device)
        seqs = []
        hist = [[] for _ in range(B)]
        alphas_list = []
        prev = None

        for _ in range(max_len):
            ctx, alpha = self.attention(img_f, h)
            alphas_list.append(alpha.detach())
            h, c = self.lstm_cell(torch.cat([self.embedding(cur), ctx], 1), (h, c))
            logits = self.fc(h)
            logits = _apply_rep_penalty(logits, hist, prev, rep_penalty, pad_idx, eos_idx, sos_idx)

            next_tok = logits.argmax(1)
            next_tok = torch.where(done, torch.full_like(next_tok, pad_idx), next_tok)
            done = done | (next_tok == eos_idx)

            seqs.append(next_tok.unsqueeze(1))
            for i in range(B):
                t = next_tok[i].item()
                if t not in (pad_idx, eos_idx, sos_idx):
                    hist[i].append(t)
            prev = next_tok
            cur = next_tok

            if done.all(): break

        result = torch.cat(seqs, 1)
        if return_alphas:
            return result, torch.stack(alphas_list, 1)
        return result

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, encoder_dim=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.memory_proj = nn.Linear(encoder_dim, hidden_dim) if encoder_dim != hidden_dim else nn.Identity()
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self._init_pos_encoding(512)

    def _init_pos_encoding(self, max_len):
        pe = torch.zeros(max_len, self.hidden_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * (-np.log(10000.0) / self.hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, img_f, q_h, q_c, ans, tfr=0.5):
        B, T = ans.size()
        memory = self.memory_proj(img_f)
        q_mem = q_h.unsqueeze(1)
        memory = torch.cat([q_mem, memory], dim=1)

        emb = self.embedding(ans) + self.pe[:, :T]
        causal_mask = self._causal_mask(T, ans.device)
        out = self.transformer_decoder(emb, memory, tgt_mask=causal_mask)
        return self.fc(out), None

    def generate(self, img_f, q_h, q_c, max_len, sos_idx, eos_idx, pad_idx=0, rep_penalty=3.0, return_alphas=False):
        B = img_f.size(0)
        memory = self.memory_proj(img_f)
        q_mem = q_h.unsqueeze(1)
        memory = torch.cat([q_mem, memory], dim=1)
        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=img_f.device)
        done = torch.zeros(B, dtype=torch.bool, device=img_f.device)
        hist = [[] for _ in range(B)]
        prev = None
        seqs = []

        for step in range(max_len):
            T = generated.size(1)
            emb = self.embedding(generated) + self.pe[:, :T]
            causal_mask = self._causal_mask(T, img_f.device)
            out = self.transformer_decoder(emb, memory, tgt_mask=causal_mask)
            logits = self.fc(out[:, -1, :])
            logits = _apply_rep_penalty(logits, hist, prev, rep_penalty, pad_idx, eos_idx, sos_idx)
            
            next_tok = logits.argmax(1)
            next_tok = torch.where(done, torch.full_like(next_tok, pad_idx), next_tok)
            done = done | (next_tok == eos_idx)
            seqs.append(next_tok.unsqueeze(1))
            
            for i in range(B):
                t = next_tok[i].item()
                if t not in (pad_idx, eos_idx, sos_idx):
                    hist[i].append(t)
            prev = next_tok
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            
            if done.all():
                break

        result = torch.cat(seqs, 1)
        if return_alphas:
            return result, None
        return result

class VQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, attention_dim=512, decoder_type='lstm_att', pretrained_cnn=False):
        super().__init__()
        self.encoder_img = ImageEncoder(hidden_dim=hidden_dim, pretrained=pretrained_cnn, trainable_cnn=False)
        self.encoder_q = QuestionEncoder(vocab_size, embed_dim, hidden_dim)
        self.decoder_type = decoder_type
        enc_dim = self.encoder_img.feature_dim
        
        if decoder_type == 'lstm_att':
            self.decoder = DecoderWithAttention(vocab_size, embed_dim, hidden_dim, enc_dim, attention_dim)
        elif decoder_type == 'transformer':
            self.decoder = TransformerDecoder(vocab_size, embed_dim, hidden_dim, enc_dim)
        else:
            raise ValueError("Unsupported decoder_type. Use 'lstm_att' for A1 or 'transformer' for A2.")

    def forward(self, images, questions, answers, bboxes=None, bbox_mask=None, teacher_forcing_ratio=0.5):
        f = self.encoder_img(images, bboxes, bbox_mask)
        q_h, q_c = self.encoder_q(questions)
        result = self.decoder(f, q_h, q_c, answers, tfr=teacher_forcing_ratio)
        if isinstance(result, tuple):
            return result
        return result, None

# ---------------------------------------------------------
# 4. BENCHMARK LOGIC (Hướng A)
# ---------------------------------------------------------

def evaluate_model(model, dataloader, dataset, device):
    """Evaluate the custom model using EM, BLEU, ROUGE, METEOR, and Recall."""
    model.eval()
    tok_correct = tok_total = exact_match = n_samples = 0
    bleu_list, rouge_list, meteor_list, recall_list = [], [], [], []
    
    smoother = SmoothingFunction().method4
    rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    pad_idx = dataset.pad_idx

    with torch.no_grad():
        for (images, questions, ans_in, ans_out, bboxes, bbox_mask) in tqdm(dataloader, desc='Benchmarking', leave=False):
            images = images.to(device)
            questions = questions.to(device)
            ans_out = ans_out.to(device)
            bboxes = bboxes.to(device)
            bbox_mask = bbox_mask.to(device)

            feat = model.encoder_img(images, bboxes, bbox_mask)
            q_h, q_c = model.encoder_q(questions)

            gen_out = model.decoder.generate(
                feat, q_h, q_c,
                max_len=dataset.max_seq_len,
                sos_idx=dataset.sos_idx,
                eos_idx=dataset.eos_idx,
                pad_idx=pad_idx,
                rep_penalty=3.0
            )
            gen = gen_out[0] if isinstance(gen_out, tuple) else gen_out

            for i in range(images.size(0)):
                pred_w = ids_to_words(gen[i].cpu().tolist(), dataset)
                tgt_w = ids_to_words(ans_out[i].cpu().tolist(), dataset)
                pred_str = ' '.join(pred_w)
                tgt_str = ' '.join(tgt_w)

                if pred_w == tgt_w: 
                    exact_match += 1

                bleu_list.append(sentence_bleu([tgt_w], pred_w, smoothing_function=smoother))
                rout = rscorer.score(tgt_str, pred_str)
                rouge_list.append(rout['rougeL'].fmeasure)
                
                if tgt_w:
                    meteor_list.append(meteor_score([tgt_w], pred_w))
                else:
                    meteor_list.append(0.0)

                recall_list.append(len(set(pred_w) & set(tgt_w)) / len(set(tgt_w)) if tgt_w else 0.0)
                n_samples += 1

    n = max(n_samples, 1)
    results = {
        'exact_match': exact_match / n,
        'bleu': sum(bleu_list) / n,
        'rouge_l': sum(rouge_list) / n,
        'meteor': sum(meteor_list) / n,
        'recall': sum(recall_list) / n,
    }
    return results

# ---------------------------------------------------------
# 5. QWEN2-VL EVALUATION LOGIC (Hướng B)
# ---------------------------------------------------------
class Qwen2VLVQA_Inference:
    """Wrapper for Qwen2-VL inference supporting Zero-shot (B1) and LoRA (B2)."""
    def __init__(self, lora_path=None, device='cuda'):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("Please install transformers and qwen_vl_utils to use Hướng B (Qwen2-VL).")

        self.device = device
        self.model_id = 'Qwen/Qwen2-VL-2B-Instruct'
        self.process_vision_info = process_vision_info
        
        print(f'Loading Qwen2-VL processor from {self.model_id} ...')
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=64 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

        print(f'Loading Qwen2-VL-2B base model...')
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

        if lora_path and os.path.exists(lora_path):
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError("Please install peft to load LoRA weights.")
            print(f'Loading LoRA adapter from {lora_path}...')
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print('No LoRA adapter provided. Running in Zero-shot mode.')
            self.model = base_model
            
        self.model.eval()

    def _build_messages(self, image, question: str) -> list:
        vietnamese_prompt = f"Bạn là chuyên gia trong lĩnh vực nhận diện hình ảnh múa lân. Vui lòng trả lời câu hỏi sau về bức ảnh bằng tiếng Việt: {question}"
        
        return [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text',  'text': vietnamese_prompt},
                ],
            }
        ]

    def predict(self, image_path: str, question: str, max_new_tokens: int = 50) -> str:
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            return 'Image error'

        messages = self._build_messages(image, question)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def evaluate_on_dataset(self, val_json: str, img_dir: str) -> dict:
        with open(val_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        smoother = SmoothingFunction().method4
        rscorer  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        bleu_list, rouge_list, meteor_list, recall_list = [], [], [], []
        exact_match = 0

        for item in tqdm(data, desc='Benchmarking Qwen2-VL', leave=False):
            img_path    = os.path.join(img_dir, item['image_path'])
            question    = item['question']
            gt_answer   = normalize_special_phrases(item['answer'])
            pred_answer = normalize_special_phrases(self.predict(img_path, question))

            ref_w  = gt_answer.split()
            pred_w = pred_answer.split()

            if pred_w == ref_w:
                exact_match += 1
                
            bleu_list.append(sentence_bleu([ref_w], pred_w, smoothing_function=smoother))
            rout = rscorer.score(gt_answer, pred_answer)
            rouge_list.append(rout['rougeL'].fmeasure)
            meteor_list.append(meteor_score([ref_w], pred_w) if ref_w else 0.0)
            recall_list.append(len(set(pred_w) & set(ref_w)) / len(set(ref_w)) if ref_w else 0.0)

        n = max(len(data), 1)
        return {
            'exact_match': exact_match / n,
            'bleu': sum(bleu_list) / n,
            'rouge_l': sum(rouge_list) / n,
            'meteor': sum(meteor_list) / n,
            'recall': sum(recall_list) / n,
        }

# ---------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VQA Benchmark for Models A1, A2, B1, or B2.')
    parser.add_argument('--model_type', type=str, required=True, choices=['a1', 'a2', 'b1', 'b2'], 
                        help='Specify "a1" (LSTM+Att), "a2" (Transformer), "b1" (Qwen Zero-shot), or "b2" (Qwen LoRA)')
    
    parser.add_argument('--weight_dir', default='./weight', type=str, help='Directory containing the model weights.')
    parser.add_argument('--test_json', default='./test.json', type=str, help='Path to test.json file')
    parser.add_argument('--vocab_json', default='./word_vocab.json', type=str, help='Path to word_vocab.json file')
    parser.add_argument('--images_dir', default='./images', type=str, help='Path to the images directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation (Applies to A1/A2)')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
 
    print('Drive path:', args.images_dir)
    # print('Files found:', os.listdir(args.images_dir))

    metrics = None

    # ---------------------------------------------------------
    # ROUTE 1: HƯỚNG A (Custom CNN + RNN/Transformer)
    # ---------------------------------------------------------
    if args.model_type in ['a1', 'a2']:
        if args.model_type == 'a1':
            weight_path = os.path.join(args.weight_dir, 'best_vqa_model_A1_pretrain_lstm_attention.pth')  
        else:
            weight_path = os.path.join(args.weight_dir, 'best_vqa_model_A2_pretrain_transformer.pth')

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found at {weight_path}")

        print('Loading dataset...')
        test_dataset = VQADataset(
            json_path=args.test_json, 
            vocab_path=args.vocab_json, 
            img_dir=args.images_dir
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f'Loading state dictionary from {weight_path}...')
        state_dict = torch.load(weight_path, map_location=device)

        ckpt_vocab_size = state_dict['decoder.fc.weight'].shape[0]
        print(f'Overriding vocab_size: {ckpt_vocab_size} (Extracted from checkpoint)')

        decoder_type = 'lstm_att' if args.model_type == 'a1' else 'transformer'

        print(f'Initializing Model {args.model_type.upper()} ({decoder_type})...')
        model = VQAModel(
            vocab_size=ckpt_vocab_size, 
            decoder_type=decoder_type, 
            pretrained_cnn=False
        ).to(device)

        has_layer_norm = any("encoder_q.layer_norm" in k for k in state_dict.keys())
        if not has_layer_norm:
            print('Older checkpoint detected. Disabling LayerNorm in QuestionEncoder.')
            model.encoder_q.layer_norm = nn.Identity()

        model.load_state_dict(state_dict, strict=False)
        print('Starting benchmark evaluation...')
        metrics = evaluate_model(model, test_loader, test_dataset, device)


    elif args.model_type in ['b1', 'b2']:
        lora_path = os.path.join(args.weight_dir, 'qwen2') if args.model_type == 'b2' else None
        
        if args.model_type == 'b2' and not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA directory not found at {lora_path}")

        print(f'Initializing Model {args.model_type.upper()} (Qwen2-VL)...')
        qwen_evaluator = Qwen2VLVQA_Inference(lora_path=lora_path, device=device)
        print('Starting benchmark evaluation...')
        metrics = qwen_evaluator.evaluate_on_dataset(args.test_json, args.images_dir)

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    if metrics:
        print('\n========================================')
        print(f'BENCHMARK RESULTS FOR MODEL {args.model_type.upper()}')
        print('========================================')
        print(f"Exact Match : {metrics['exact_match']:.4f}")
        print(f"BLEU-4      : {metrics['bleu']:.4f}")
        print(f"ROUGE-L     : {metrics['rouge_l']:.4f}")
        print(f"METEOR      : {metrics['meteor']:.4f}")
        print(f"Token Recall: {metrics['recall']:.4f}")
        print('========================================\n')