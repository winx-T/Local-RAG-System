# 🧩 Local RAG System with Ollama & ChromaDB

A lightweight, **fully local** Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about your own **PDFs or text files**—powered entirely by **open-source models** and **zero cloud dependencies**.

- **Embeddings**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via Hugging Face `sentence-transformers`  
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) with persistent storage (`./rag_db`)  
- **Language Model**: [`llama3.2:3b`](https://ollama.com/library/llama3.2) served locally via **[Ollama](https://ollama.com/)**  
- **Supported Inputs**: `.pdf` and `.txt` documents  

> 🔒 **100% offline** — your documents and queries never leave your machine.

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **[Ollama](https://ollama.com/)** installed and running
- Download the LLM:
  ```bash
  ollama pull llama3.2:3b
  ```

> 💡 Ollama runs automatically in the background after installation on macOS/Windows. On Linux, you may need to start it manually.

---

### 2. Clone & Install

```bash
git clone https://github.com/your-username/local-rag-ollama.git
cd local-rag-ollama
pip install -r requirements.txt
```

> ⚠️ First run will download the embedding model (~80 MB). This happens only once.

---

### 3. Run the Application

```bash
python rag_app.py
```

---

### 4. Use the Interactive Menu

1. **Add a document**  
   → Supports `.pdf` and `.txt` (e.g., `cloud_computing.txt`)  
2. **Ask a question**  
   → The system retrieves relevant text and generates an answer using Llama 3.2  
3. **Exit** when done  

#### ✅ Example Workflow

```text
=== 🧩 Local RAG System (Hugging Face embeddings + Ollama Llama3) ===
1️⃣ Add document (.txt or .pdf)
2️⃣ Ask a question
3️⃣ Exit

Choose an option (1/2/3): 1
Enter file path: C:/Users/hp/Desktop/cloud_computing.txt
📚 Adding 6 chunks from cloud_computing.txt
✅ Document embedded and stored (6/6 chunks added).

Choose an option (1/2/3): 2
Enter your question: What are the main types of cloud computing?

🧠 Final Answer:

According to the context, there are three main types of cloud computing services:

1. Infrastructure as a Service (IaaS)
2. Platform as a Service (PaaS)
3. Software as a Service (SaaS)
```

---

## 🧠 How It Works

1. **Ingest**  
   Your document is split into small chunks (~300 characters).
2. **Embed**  
   Each chunk is converted into a vector using `all-MiniLM-L6-v2`.
3. **Store**  
   Chunks and embeddings are saved in a persistent ChromaDB collection.
4. **Retrieve**  
   Your question is embedded and matched against stored chunks (top 3 results).
5. **Generate**  
   Relevant context + your question → sent to Ollama → Llama 3.2 returns a grounded answer.

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Core packages include:

- `chromadb`
- `PyPDF2`
- `sentence-transformers`
- `requests`

Install with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Notes & Tips

- **Chunk size** is set to 300 characters (adjustable in `chunk_text()`).
- The vector database persists in `./rag_db`—no data is lost between runs.
- If Ollama isn’t running, you’ll see a connection error. Make sure it’s active!
- For large documents, ingestion may take a few seconds.

---

## 📜 License

Distributed under the **MIT License**.  
Feel free to use, modify, and share—personally or commercially.

---

> 💬 **Built with ❤️ for privacy-conscious developers, researchers, and tinkerers.**  
> Inspired by the power of open models and local AI.

---
