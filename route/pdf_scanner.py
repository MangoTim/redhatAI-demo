from flask import Blueprint, request, jsonify
import fitz  # PyMuPDF
import os, glob, torch
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModel
from flasgger import swag_from
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from api_docs.pdf_scanner_docs import upload_pdf_doc, ask_pdf_doc, remove_pdf_doc
from modules import get_model, log_message, load_pdf_prompt_template, get_db_connection, clean_model_output, remove_speculative_intro, extract_answer

pdf_scanner = Blueprint("pdf_scanner", __name__)

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

@pdf_scanner.route("/upload_pdf", methods=["POST"])
@swag_from(upload_pdf_doc)
def upload_pdf():
    try:
        file = request.files.get("pdf")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        os.makedirs("uploads", exist_ok=True)
        clean_folder("uploads")

        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
        print(f"File saved to: {file_path}", flush=True)

        text = extract_pdf_text(file_path)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM pdf_documents")
        cur.execute("INSERT INTO pdf_documents (title, content, uploaded_at) VALUES (%s, %s, NOW())", (filename, text))
        conn.commit()
        cur.close()
        conn.close()

        # Chunk and embed
        global pdf_chunks, pdf_embeddings
        pdf_chunks = chunk_text(text)
        pdf_embeddings = embed_chunks(pdf_chunks)

        print(f"Inserted and embedded PDF: {filename}", flush=True)
        return jsonify({"status": "scanned", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@pdf_scanner.route("/ask_pdf", methods=["POST"])
@swag_from(ask_pdf_doc)
def ask_pdf():
    try:
        data = request.get_json()
        print(f"Received payload: {data}", flush=True)

        question = (data.get("question") or data.get("message", "")).strip()
        model_key = data.get("model")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        model, is_chat_model, has_tokenizer = get_model(model_key)
        if not model:
            return jsonify({"error": f"Unsupported model: {model_key}"}), 400

        # Retrieve top chunks
        relevant_chunks = retrieve_relevant_chunks(question)
        print("\n🔍 Retrieved Chunks Used for Prompt:\n", flush=True)
        for i, chunk in enumerate(relevant_chunks):
            print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}", flush=True)
        if not relevant_chunks:
            return jsonify({"error": "No PDF content available"}), 404

        context = "\n\n".join(relevant_chunks)
        prompt_template = load_pdf_prompt_template()
        prompt_text = prompt_template.format(context=context, question=question)

        print(f"Prompt sent to model:\n{prompt_text}", flush=True)
        result = model(prompt_text, max_new_tokens=1024)

        raw_output = (
            result[0]["generated_text"].strip()
            if isinstance(result, list) and "generated_text" in result[0]
            else str(result).strip()
        )

        reply = extract_answer(raw_output)
        reply = remove_speculative_intro(reply)
        answer = clean_model_output(reply, prompt_text=prompt_text, similarity_threshold=0.9)

        log_message("user", question)
        log_message("assistant", answer)

        print(f"Returning answer: {answer}", flush=True)
        return jsonify({"response": answer})
    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 500

@pdf_scanner.route("/remove_pdf", methods=["POST"])
@swag_from(remove_pdf_doc)
def remove_pdf():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM pdf_documents")
        conn.commit()
        cur.close()
        conn.close()

        for file in glob.glob("uploads/*.pdf"):
            os.remove(file)

        global pdf_chunks, pdf_embeddings
        pdf_chunks, pdf_embeddings = [], []

        return jsonify({"message": "PDF removed from database and memory."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500