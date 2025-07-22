import os
import json
import fitz  # PyMuPDF
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel
from safetensors.torch import load_file as safetensors_load_file

# ---------------------------------------------------------
# Locate model root (relative to this file)
# ---------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(_THIS_DIR, "models", "all-MiniLM-L6-v2")

def _resolve_transformer_dir(root: str) -> str:
    """
    Return the directory that contains config + weights.
    Supports SentenceTransformers layout & HF layout.
    """
    # direct HF-style
    if os.path.exists(os.path.join(root, "config.json")) and (
        os.path.exists(os.path.join(root, "pytorch_model.bin")) or
        os.path.exists(os.path.join(root, "model.safetensors"))
    ):
        return root

    # SentenceTransformers submodule layout (0_Transformer)
    st_sub = os.path.join(root, "0_Transformer")
    if os.path.exists(os.path.join(st_sub, "config.json")) and (
        os.path.exists(os.path.join(st_sub, "pytorch_model.bin")) or
        os.path.exists(os.path.join(st_sub, "model.safetensors"))
    ):
        return st_sub

    raise RuntimeError(
        "Could not find transformer weights.\n"
        f"Checked: {root} and {st_sub}\n"
        "Make sure the MiniLM model folder is copied into the image."
    )

MODEL_DIR = _resolve_transformer_dir(MODEL_ROOT)

# ---------------------------------------------------------
# Offline model + tokenizer load (manual weight load if needed)
# ---------------------------------------------------------
def _load_model_and_tokenizer(model_dir: str):
    # tokenizer loads fine from HF layout or 0_Transformer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_config(cfg)  # empty weights

    # Determine weight file
    weights_bin = os.path.join(model_dir, "pytorch_model.bin")
    weights_sft = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(weights_bin):
        state_dict = torch.load(weights_bin, map_location="cpu")
    elif os.path.exists(weights_sft):
        state_dict = safetensors_load_file(weights_sft, device="cpu")
    else:
        raise RuntimeError(f"No model weights found in {model_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading model: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading model: {len(unexpected)}")

    model.eval()
    model.to(torch.device("cpu"))
    return tokenizer, model

tokenizer, model = _load_model_and_tokenizer(MODEL_DIR)
DEVICE = torch.device("cpu")

# ---------------------------------------------------------
# Heading example bank
# ---------------------------------------------------------
HEADING_EXAMPLES = {
    "H1": [
        "Introduction", "Background", "Conclusion", "Overview",
        "Executive Summary", "Related Work", "Discussion", "Summary"
    ],
    "H2": [
        "Problem Statement", "Objectives", "Scope", "Methodology",
        "Limitations", "Future Work", "Results", "Data Analysis"
    ],
    "H3": [
        "Example", "Details", "Notes", "Method Detail",
        "Step-by-Step", "Additional Information"
    ],
}

# Precompute example embeddings
_example_texts = []
_example_labels = []
for lbl, arr in HEADING_EXAMPLES.items():
    for t in arr:
        _example_texts.append(t)
        _example_labels.append(lbl)

def _embed_texts(texts, max_length=128):
    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    token_embeddings = outputs.last_hidden_state  # [B,S,H]
    mask = encoded["attention_mask"].unsqueeze(-1).type_as(token_embeddings)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    mean_embeddings = summed / counts
    mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
    return mean_embeddings.cpu().numpy()

_example_embeds = _embed_texts(_example_texts)  # [N,H]

def _cosine_sim(v1, mat):
    num = mat @ v1
    denom = (np.linalg.norm(mat, axis=1) * np.linalg.norm(v1) + 1e-9)
    return num / denom

def semantic_heading_match(text, threshold=0.60):
    if not text or len(text.strip()) < 3:
        return None
    v = _embed_texts([text])[0]
    sims = _cosine_sim(v, _example_embeds)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score < threshold:
        return None
    return _example_labels[best_idx]

# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------
def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                line_text = ""
                font_sizes = []
                fonts = []
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if t:
                        line_text += t + " "
                        font_sizes.append(span["size"])
                        fonts.append(span["font"])
                if not line_text.strip():
                    continue
                avg_font_size = sum(font_sizes) / len(font_sizes)
                lines.append({
                    "text": line_text.strip(),
                    "font_size": avg_font_size,
                    "font": fonts[0] if fonts else "",
                    "page": page_num
                })
    return lines

# ---------------------------------------------------------
# Hybrid classify
# ---------------------------------------------------------
def classify_headings(lines):
    font_sizes = [round(line["font_size"], 1) for line in lines]
    sorted_sizes = sorted(set(font_sizes), reverse=True)

    font_to_level = {}
    if len(sorted_sizes) >= 4:
        font_to_level[sorted_sizes[0]] = "Title"
        font_to_level[sorted_sizes[1]] = "H1"
        font_to_level[sorted_sizes[2]] = "H2"
        font_to_level[sorted_sizes[3]] = "H3"
    elif len(sorted_sizes) >= 3:
        font_to_level[sorted_sizes[0]] = "Title"
        font_to_level[sorted_sizes[1]] = "H1"
        font_to_level[sorted_sizes[2]] = "H2"
    elif len(sorted_sizes) >= 2:
        font_to_level[sorted_sizes[0]] = "Title"
        font_to_level[sorted_sizes[1]] = "H1"
    elif len(sorted_sizes) >= 1:
        font_to_level[sorted_sizes[0]] = "Title"

    title = None
    outline = []

    for line in lines:
        text = line["text"]
        size = round(line["font_size"], 1)
        level = font_to_level.get(size)

        sem_level = None
        if not level:
            if len(text) <= 12 or text.isupper() or text.istitle():
                sem_level = semantic_heading_match(text)

        final_level = level or sem_level

        if final_level == "Title" and title is None:
            title = text
        elif final_level in ["H1", "H2", "H3"]:
            outline.append({
                "level": final_level,
                "text": text,
                "page": line["page"]
            })
    return title, outline

# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------
def save_output_json(title, outline, output_path):
    data = {
        "title": title or "Unknown Title",
        "outline": outline
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
