# import fitz  # PyMuPDF
# from collections import Counter
# import json

# def extract_text_blocks(pdf_path):
#     doc = fitz.open(pdf_path)
#     lines = []

#     for page_num, page in enumerate(doc, start=1):
#         blocks = page.get_text("dict")["blocks"]
#         for block in blocks:
#             for line in block.get("lines", []):
#                 line_text = ""
#                 font_sizes = []
#                 fonts = []

#                 for span in line["spans"]:
#                     line_text += span["text"].strip() + " "
#                     font_sizes.append(span["size"])
#                     fonts.append(span["font"])

#                 if not line_text.strip():
#                     continue

#                 avg_font_size = sum(font_sizes) / len(font_sizes)
#                 lines.append({
#                     "text": line_text.strip(),
#                     "font_size": avg_font_size,
#                     "font": fonts[0],
#                     "page": page_num
#                 })
#     return lines


# def classify_headings(lines):
#     font_sizes = [round(line["font_size"], 1) for line in lines]
#     sorted_sizes = sorted(set(font_sizes), reverse=True)

#     font_to_level = {}
#     if len(sorted_sizes) >= 4:
#         font_to_level[sorted_sizes[0]] = "Title"
#         font_to_level[sorted_sizes[1]] = "H1"
#         font_to_level[sorted_sizes[2]] = "H2"
#         font_to_level[sorted_sizes[3]] = "H3"
#     elif len(sorted_sizes) >= 3:
#         font_to_level[sorted_sizes[0]] = "Title"
#         font_to_level[sorted_sizes[1]] = "H1"
#         font_to_level[sorted_sizes[2]] = "H2"
#     elif len(sorted_sizes) >= 2:
#         font_to_level[sorted_sizes[0]] = "Title"
#         font_to_level[sorted_sizes[1]] = "H1"
#     elif len(sorted_sizes) >= 1:
#         font_to_level[sorted_sizes[0]] = "Title"

#     title = None
#     outline = []

#     for line in lines:
#         size = round(line["font_size"], 1)
#         level = font_to_level.get(size)

#         if level == "Title" and not title:
#             title = line["text"]
#         elif level in ["H1", "H2", "H3"]:
#             outline.append({
#                 "level": level,
#                 "text": line["text"],
#                 "page": line["page"]
#             })

#     return title, outline


# def save_output_json(title, outline, output_path):
#     output = {
#         "title": title or "Unknown Title",
#         "outline": outline
#     }

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2)


import fitz  # PyMuPDF
import json
from collections import Counter
from sentence_transformers import SentenceTransformer, util

# Load MiniLM model
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define example headings for semantic comparison
HEADING_EXAMPLES = {
    "H1": ["Introduction", "Background", "Conclusion", "Overview", "Summary", "Related Work", "Discussion"],
    "H2": ["Problem Statement", "Objectives", "Scope", "Limitations", "Future Work", "Results"],
    "H3": ["Subsection A", "Subsection B", "Method Detail", "Example", "Step-by-Step"]
}

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

                for span in line["spans"]:
                    line_text += span["text"].strip() + " "
                    font_sizes.append(span["size"])
                    fonts.append(span["font"])

                if not line_text.strip():
                    continue

                avg_font_size = sum(font_sizes) / len(font_sizes)
                lines.append({
                    "text": line_text.strip(),
                    "font_size": avg_font_size,
                    "font": fonts[0],
                    "page": page_num
                })
    return lines

def semantic_heading_match(text):
    if len(text) < 3:
        return None

    text_vec = minilm_model.encode(text, convert_to_tensor=True)

    best_match = None
    best_score = 0.6  # You can tune this

    for level, examples in HEADING_EXAMPLES.items():
        for example in examples:
            example_vec = minilm_model.encode(example, convert_to_tensor=True)
            score = util.pytorch_cos_sim(text_vec, example_vec).item()

            if score > best_score:
                best_score = score
                best_match = level

    return best_match

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
        size = round(line["font_size"], 1)
        level = font_to_level.get(size)

        # Use semantic fallback or override
        sem_level = semantic_heading_match(line["text"])
        final_level = level or sem_level

        if final_level == "Title" and not title:
            title = line["text"]
        elif final_level in ["H1", "H2", "H3"]:
            outline.append({
                "level": final_level,
                "text": line["text"],
                "page": line["page"]
            })

    return title, outline

def save_output_json(title, outline, output_path):
    output = {
        "title": title or "Unknown Title",
        "outline": outline
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
