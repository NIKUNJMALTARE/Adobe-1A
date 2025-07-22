from transformers import AutoModel, AutoTokenizer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_DIR = os.path.join("models", "all-MiniLM-L6-v2")

print("Downloading MiniLM model... (one time only)")
os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print(f"Saving to {SAVE_DIR} ...")
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("Download completed. Model saved locally!")
