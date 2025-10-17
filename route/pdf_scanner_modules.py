import fitz  # PyMuPDF
import os, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

model_path = "ibm-granite/granite-embedding-125m-english"
embedding_tokenizer = AutoTokenizer.from_pretrained(model_path)
embedding_model = AutoModel.from_pretrained(model_path)

pdf_chunks = []
pdf_embeddings = []

def get_embedding(text: str) -> torch.Tensor:
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def embed_chunks(chunks):
    return [get_embedding(chunk) for chunk in chunks]

def retrieve_relevant_chunks(question, top_k=3):
    if not pdf_chunks or not pdf_embeddings:
        return []
    q_vec = get_embedding(question).unsqueeze(0).numpy()
    chunk_vecs = torch.stack(pdf_embeddings).numpy()
    scores = cosine_similarity(q_vec, chunk_vecs)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [pdf_chunks[i] for i in top_indices]

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = "".join(page.get_text() for page in doc)
    metadata = doc.metadata
    meta_info = "\n".join(f"{k}: {v}" for k, v in metadata.items() if v)
    return f"{meta_info}\n\n{text}"

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}", flush=True)