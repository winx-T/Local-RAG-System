import os
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Hugging Face embedding model
GEN_MODEL = "llama3.2:3b"                 # Ollama LLM for generation
DB_DIR = "./rag_db"

# ---------- LOAD EMBEDDING MODEL ----------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ---------- FUNCTIONS ----------

def get_embedding(text: str):
    """Get embedding using Hugging Face model."""
    if not text.strip():
        return None
    return embed_model.encode(text).tolist()  # convert to list for ChromaDB

def get_llm_response(prompt: str):
    """Generate response from Ollama Llama3, handling streaming JSON output."""
    import requests, json

    url = "http://localhost:11434/api/generate"
    data = {"model": GEN_MODEL, "prompt": prompt}

    try:
        response = requests.post(url, json=data, stream=True, timeout=120)
        response.raise_for_status()

        final_text = ""
        # Ollama streams JSON objects per line
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "response" in obj and obj["response"]:
                    final_text += obj["response"]
            except json.JSONDecodeError:
                continue

        return final_text.strip()

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Error generating response."


def read_pdf(file_path):
    """Extract text from PDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=300):
    """Split text into chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ---------- DATABASE SETUP ----------
chroma_client = chromadb.Client(Settings(
    persist_directory=DB_DIR
))
collection = chroma_client.get_or_create_collection("knowledge_base")

# ---------- DOCUMENT INGESTION ----------
def add_document(file_path):
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    if file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"❌ Unsupported file type: {file_path}")
        return

    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    chunks = chunk_text(text)
    print(f"📚 Adding {len(chunks)} chunks from {os.path.basename(file_path)}")

    valid_chunks = 0
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        if emb is None:
            continue
        try:
            collection.add(
                ids=[f"{os.path.basename(file_path)}-{i}"],
                documents=[chunk],
                embeddings=[emb]
            )
            valid_chunks += 1
        except Exception as e:
            print(f"⚠️ Skipping chunk {i} due to error: {e}")

    print(f"✅ Document embedded and stored ({valid_chunks}/{len(chunks)} chunks added).")

# ---------- QUERYING ----------
def query_rag(question):
    if collection.count() == 0:
        print("⚠️ The database is empty! Add documents first.")
        return

    query_emb = get_embedding(question)
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    if not results["documents"]:
        print("⚠️ No relevant documents found.")
        return

    context = "\n".join(results["documents"][0])
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = get_llm_response(prompt)
    print("\n🧠 Final Answer:\n")
    print(answer)

# ---------- MAIN LOOP ----------
if __name__ == "__main__":
    print("=== 🧩 Local RAG System (Hugging Face embeddings + Ollama Llama3) ===")
    print("1️⃣ Add document (.txt or .pdf)")
    print("2️⃣ Ask a question")
    print("3️⃣ Exit")

    while True:
        choice = input("\nChoose an option (1/2/3): ").strip()
        if choice == "1":
            raw_path = input("Enter file path: ").strip().strip('"')
            path = os.path.normpath(raw_path)
            add_document(path)
        elif choice == "2":
            question = input("Enter your question: ").strip()
            query_rag(question)
        elif choice == "3":
            print("👋 Exiting.")
            break
        else:
            print("❌ Invalid choice.")
