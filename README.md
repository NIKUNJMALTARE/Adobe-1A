#Adobe Hackathon Challenge 1A

This project addresses **Adobe Hackathon Challenge 1A**, which involves extracting structured semantic outlines from PDF documents under strict constraints:

- **Model size ≤ 200MB**
- **No internet access allowed**
- **Must run on CPU only (amd64 architecture)**
- **Execution time ≤ 10 seconds for 50-page PDFs**

The system outputs a structured `JSON` that includes the **document title** and an **outline** of sections (H1, H2, H3, etc.) based on layout and semantics.

---

#Approach

This solution follows a **hybrid layout-semantic approach**, meaning it combines traditional layout-based cues (like font size and font type) with semantic understanding using a transformer model.

#Steps in the Pipeline

1. **PDF Layout Parsing**  
   - Uses `PyMuPDF` to extract raw lines of text along with font sizes, font names, and page numbers.
   - Cleans and filters line-level text data.

2. **Semantic Heading Detection**  
   - A locally stored model encodes each line to a semantic embedding.
   - It compares these embeddings to predefined heading examples (like "Introduction", "Background") using cosine similarity.
   - A confidence score determines if a line is semantically close to an H1/H2/H3/H4 heading.

3. **Hybrid Logic to Classify Headings**  
   - Combines:
     - Font size and font type (layout)
     - Cosine similarity score with example headings (semantic)
   - This ensures better accuracy in documents with inconsistent layouts.

4. **Final Output**  
   - Generates a JSON with:
     - `"title"` (detected from top of the document)
     - `"outline"` array containing each heading’s level (H1–H4), text, and page number.

---

#Models & Libraries Used

#Model

- `all-MiniLM-L6-v2`
  - Model size: ~80MB (within constraint)
  - Used for sentence embeddings to semantically classify headings
  - Fully supports **offline + CPU-only execution**

#Libraries Used (in `requirements.txt`)

numpy==1.26.4
PyMuPDF==1.23.6
torch==2.2.2+cpu
transformers==4.41.2
tokenizers==0.19.1
huggingface_hub==0.24.0
safetensors==0.4.3
sentence-transformers==2.7.0
scikit-learn==1.4.0

#How to Build and Run the Solution

1) Clone the Repository 
2) Create a Virtual Environment: python -m venv venv (command)
3) For activating virtual emvironment: venv\Scripts\activate (activate command)
4) Install All Dependencies: pip install -r requirements.txt (command)
5) Download the Model Locally (One-Time): python download_model.py (command)
6) Place your .pdf files into the /input folder
7) Run the main file: python main.py (command)

#Docker Commands

1) docker build --platform linux/amd64 -t adobe1a.outlineextractor .
2) docker run --rm -v ${PWD}\input:/app/input:ro -v ${PWD}\output\adobe1a:/app/output --network none adobe1a.outlineextractor
