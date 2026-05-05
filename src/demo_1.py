
import os
import json
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import re

# Import classes and functions from your provided script
# Ensure the script is saved as 'vqa_core.py' in the same directory
try:
    from benchmark import VQAModel, Qwen2VLVQA_Inference, ids_to_words, normalize_special_phrases
except ImportError:
    st.error("Please save your original script as 'vqa_core.py' in the same folder as this app.")
    st.stop()

# Set up the web page configuration
st.set_page_config(page_title="VQA All Models Demo", layout="wide")

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS & MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource
def load_vocab(vocab_path):
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_vocab = json.load(f)
    return word_vocab

@st.cache_resource
def load_model_a(model_type, weight_path, vocab_path, device):
    """Load custom architecture models (A1, A2, or A3)."""
    word_vocab = load_vocab(vocab_path)
    
    # Load checkpoint to extract vocab size
    state_dict = torch.load(weight_path, map_location=device)
    ckpt_vocab_size = state_dict['decoder.fc.weight'].shape[0]
    
    # Determine the correct decoder type based on the selected model
    if model_type == 'a1':
        decoder_type = 'lstm_att'
    elif model_type == 'a2':
        decoder_type = 'transformer'
    elif model_type == 'a3':
        # Ensure your VQAModel implementation supports this specific string
        decoder_type = 'transformer_swiglu' 
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize model
    model = VQAModel(
        vocab_size=ckpt_vocab_size, 
        decoder_type=decoder_type, 
        pretrained_cnn=False
    ).to(device)
    
    # Handle older checkpoints without LayerNorm
    has_layer_norm = any("encoder_q.layer_norm" in k for k in state_dict.keys())
    if not has_layer_norm and hasattr(model.encoder_q, 'layer_norm'):
        model.encoder_q.layer_norm = torch.nn.Identity()
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Define standard image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, word_vocab, transform

@st.cache_resource
def load_model_b(model_type, lora_path, device):
    """Load Qwen2-VL models (B1 or B2)."""
    qwen_evaluator = Qwen2VLVQA_Inference(lora_path=lora_path, device=device)
    return qwen_evaluator

def process_question_a(question, word_vocab, max_seq_len=20):
    """Tokenize and pad question for model A architectures."""
    question = normalize_special_phrases(question)
    question = re.sub(r'[^\w\s_]', ' ', question)
    question = re.sub(r'\s+', ' ', question).strip()
    words = question.split()
    
    unk_idx = word_vocab.get('<UNK>', 1)
    pad_idx = word_vocab.get('<PAD>', 0)
    
    tokens = [word_vocab.get(w, unk_idx) for w in words]
    if len(tokens) < max_seq_len:
        tokens.extend([pad_idx] * (max_seq_len - len(tokens)))
    else:
        tokens = tokens[:max_seq_len]
        
    return torch.tensor([tokens], dtype=torch.long)

def inference_a(model, image, question_text, word_vocab, transform, device):
    """Run inference for a single image and question using Model A architectures."""
    # Process image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process question
    q_tensor = process_question_a(question_text, word_vocab).to(device)
    
    # Create dummy bounding boxes for inference without explicit annotations
    max_boxes = 6
    arr = np.zeros((1, max_boxes, 5), dtype=np.float32)
    mask = np.zeros((1, max_boxes), dtype=np.float32)
    
    # Default dummy box
    arr[0, 0] = [0.0, 0.5, 0.5, 1.0, 1.0]
    mask[0, 0] = 1.0
    
    bboxes = torch.tensor(arr).to(device)
    bbox_mask = torch.tensor(mask).to(device)
    
    with torch.no_grad():
        feat = model.encoder_img(img_tensor, bboxes, bbox_mask)
        q_h, q_c = model.encoder_q(q_tensor)
        
        pad_idx = word_vocab.get('<PAD>', 0)
        sos_idx = word_vocab.get('<SOS>', 2)
        eos_idx = word_vocab.get('<EOS>', 3)
        
        gen_out = model.decoder.generate(
            feat, q_h, q_c,
            max_len=20,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            rep_penalty=3.0
        )
        
        gen = gen_out[0] if isinstance(gen_out, tuple) else gen_out
        
        # Temporary dataset object to map predicted IDs back to words
        class DummyDataset:
            def __init__(self, vocab):
                self.idx2word = {idx: word for word, idx in vocab.items()}
                self.eos_idx = eos_idx
                self.sos_idx = sos_idx
                self.pad_idx = pad_idx
                
        pred_w = ids_to_words(gen[0].cpu().tolist(), DummyDataset(word_vocab))
        
        # Join words into a single string
        answer_str = " ".join(pred_w)
        
        # Remove underscores and normalize spaces
        answer_str = answer_str.replace('_', ' ')
        answer_str = " ".join(answer_str.split())
        
        return answer_str

# ---------------------------------------------------------
# 2. WEB INTERFACE LAYOUT
# ---------------------------------------------------------

st.title("Visual Question Answering Demo")
st.markdown("Upload an image and ask a question to get answers from all available models simultaneously.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Configure paths globally
weight_dir = st.sidebar.text_input("Weight Directory Path", value="./weight")
vocab_path = st.sidebar.text_input("Vocabulary JSON Path", value="./word_vocab.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.markdown(f"**Current Device:** {str(device).upper()}")

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        question = st.text_input("Ask a question about this image:")
        submit_button = st.button("Generate Answers")

with col2:
    st.subheader("Results")
    
    if uploaded_file is not None and question and submit_button:
        # Define the 5 models to iterate over
        models_to_run = {
            "Model A1 ": "a1",
            "Model A2 ": "a2",
            "Model A3 ": "a3",
            "Model B1 ": "b1",
            "Model B2 ": "b2"
        }
        
        # Save image temporarily once for Qwen models to use
        temp_img_path = "temp_uploaded_image.jpg"
        image.save(temp_img_path)
        
        for model_name, model_key in models_to_run.items():
            st.markdown(f"#### {model_name}")
            try:
                with st.spinner(f"Running inference for {model_key.upper()}..."):
                    if model_key in ['a1', 'a2', 'a3']:
                        # Define weight file names based on the model key
                        if model_key == 'a1':
                            weight_file = 'best_vqa_model_A1_pretrain_lstm_attention.pth'
                        elif model_key == 'a2':
                            weight_file = 'best_vqa_model_A2_pretrain_transformer.pth'
                        elif model_key == 'a3':
                            weight_file = 'best_vqa_model_A3_pretrain_transformer_rmsnorm.pth' 
                            
                        full_weight_path = os.path.join(weight_dir, weight_file)
                        
                        if not os.path.exists(full_weight_path):
                            st.warning(f"Weight file not found: {full_weight_path}")
                        elif not os.path.exists(vocab_path):
                            st.warning(f"Vocabulary file not found: {vocab_path}")
                        else:
                            model, word_vocab, transform = load_model_a(model_key, full_weight_path, vocab_path, device)
                            answer = inference_a(model, image, question, word_vocab, transform, device)
                            st.success(f"**Answer:** {answer}")
                            
                    elif model_key in ['b1', 'b2']:
                        lora_path = os.path.join(weight_dir, 'qwen2') if model_key == 'b2' else None
                        
                        if model_key == 'b2' and not os.path.exists(lora_path):
                            st.warning(f"LoRA directory not found: {lora_path}")
                        else:
                            qwen_model = load_model_b(model_key, lora_path, device)
                            raw_answer = qwen_model.predict(temp_img_path, question)
                            
                            # Clean up Qwen answer as well to be consistent
                            clean_answer = str(raw_answer).replace('_', ' ')
                            clean_answer = " ".join(clean_answer.split())
                            
                            st.success(f"**Answer:** {clean_answer}")
                            
            except Exception as e:
                st.error(f"An error occurred during inference for {model_name}: {str(e)}")
            
            # Add a visual separator between model results
            st.markdown("---")
            
        # Clean up temporary file after all models have finished processing
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)