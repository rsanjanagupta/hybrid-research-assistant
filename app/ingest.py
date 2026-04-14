import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx


# -----------------------------
# 1. Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2. Read PDF
# -----------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    return text


# -----------------------------
# 3. Read DOCX
# -----------------------------
def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


# -----------------------------
# 4. Chunk text
# -----------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# -----------------------------
# 5. Load documents
# -----------------------------
def load_documents(folder_path):
    all_chunks = []

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return all_chunks

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            print(f"📄 Reading PDF: {file}")
            text = read_pdf(file_path)

        elif file.endswith(".docx"):
            print(f"📄 Reading DOCX: {file}")
            text = read_docx(file_path)

        else:
            continue

        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    return all_chunks


# -----------------------------
# 6. Create FAISS index
# -----------------------------
def create_index(chunks):
    print("🔍 Creating embeddings...")

    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


# -----------------------------
# 7. Save index
# -----------------------------
def save_index(index, chunks, folder_path):
    index_path = os.path.join(folder_path, "faiss_index.bin")
    doc_path = os.path.join(folder_path, "documents.pkl")

    faiss.write_index(index, index_path)

    with open(doc_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Index saved at {folder_path}")


# -----------------------------
# 8. 🔥 LOAD INDEX (AUTO + FIXED)
# -----------------------------
def load_user_index(user_id, base_path="data"):
    folder_path = os.path.join(base_path, user_id)

    index_path = os.path.join(folder_path, "faiss_index.bin")
    doc_path = os.path.join(folder_path, "documents.pkl")

    # 🔥 ONLY check index (prevents repeated ingestion)
    if not os.path.exists(index_path):
        print(f"⚡ No index found for user: {user_id}")
        print("⚡ Running ingestion...")

        success = ingest_documents(user_id, base_path)

        # ❌ If no docs → return signal for fallback
        if not success:
            print("⚠️ No documents → returning empty for web fallback")
            return None, []

    # ✅ Load index
    print(f"📂 Loading index for user: {user_id}")

    index = faiss.read_index(index_path)

    # Load documents safely
    if os.path.exists(doc_path):
        with open(doc_path, "rb") as f:
            documents = pickle.load(f)
    else:
        documents = []

    return index, documents


# -----------------------------
# 9. MAIN INGEST FUNCTION
# -----------------------------
def ingest_documents(user_id, base_path="data"):
    folder_path = os.path.join(base_path, user_id)

    print(f"\n📂 Ingesting for user: {user_id}")

    os.makedirs(folder_path, exist_ok=True)

    chunks = load_documents(folder_path)

    # ❌ IMPORTANT: signal failure properly
    if len(chunks) == 0:
        print("⚠️ No documents found for this user.")
        return False

    print(f"📊 Total chunks created: {len(chunks)}")

    index, chunks = create_index(chunks)
    save_index(index, chunks, folder_path)

    return True