# import os
# from extract_utils import extract_text_blocks, classify_headings, save_output_json

# def process_pdf(pdf_path, output_path):
#     print(f"Processing: {pdf_path}")
#     lines = extract_text_blocks(pdf_path)
#     title, outline = classify_headings(lines)
#     save_output_json(title, outline, output_path)
#     print(f"Saved to: {output_path}")

# if __name__ == "__main__":
#     input_dir = "./input"
#     output_dir = "./output"
#     os.makedirs(output_dir, exist_ok=True)

#     for filename in os.listdir(input_dir):
#         if filename.endswith(".pdf"):
#             pdf_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
#             process_pdf(pdf_path, output_path)

import os
from extract_utils import extract_text_blocks, classify_headings, save_output_json

def process_pdf(pdf_path, output_path):
    print(f"Processing: {pdf_path}")
    lines = extract_text_blocks(pdf_path)
    title, outline = classify_headings(lines)
    save_output_json(title, outline, output_path)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    input_dir = "./input"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
            process_pdf(pdf_path, output_path)
