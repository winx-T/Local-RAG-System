# 🧠 Local RAG System – Smart, Simple, and Powerful

A **zero-setup**, **fully local Retrieval-Augmented Generation (RAG) system** that lets you instantly ask questions about your own **PDFs, Word, Excel, PowerPoint, Text, CSV, or HTML files** — **with no cloud, no API keys, and complete privacy**.

Built for **everyone**:
- 🧩 Beginners who just want a simple, working personal AI.
- 🧠 Power users who want reranking, hybrid search, and multilingual embeddings.

---

## 🔍 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI technique that combines *information retrieval* with *text generation*.  
Instead of relying only on what a model “knows” from training, RAG allows the system to **search your documents** in real time and use that knowledge to produce accurate, grounded answers.

<p align="center">
  <img 
    src="https://cdn.prod.website-files.com/651c34ac817aad4a2e62ec1b/655664de69b30a6d00f0960c_gaJkRvUmWHsWtnAGlNtjQJYhSzHvUwZHvV7nDU3kQJ6EyEI1C4v6HRysXIw28UlXK3QT4yU0rgTD7v1cUgbl5nB71emE5vqz9Y0VlvLjg10BgaLcOvI4Zauu9AKU6EKWN5rIwIKPs8CSYd0CiX2Gg5g.png" 
    alt="🧠 Local RAG System Banner"
    style="max-width: 90%; height: auto; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);"
  >
</p>

### 🧠 How RAG Works

1. **You ask a question.**  
2. The system **retrieves** the most relevant chunks of text from your local knowledge base (PDFs, Word docs, etc.).  
3. These chunks are **combined** with your question to form a detailed prompt.  
4. The **language model** (LLM) then generates an answer — using the retrieved data as factual grounding.  
5. The answer is shown with **source references** so you know exactly where it came from.

### ✅ Why It Matters

- Prevents AI “hallucinations” by grounding answers in real documents.  
- Works with **your own data**, not just what’s in the model’s memory.  
- No retraining needed — just add documents and start asking!  
- In your local version, everything happens **offline** and **privately**.
---

## ⚙️ Installation

### 🧠 Requirements

* **Python 3.8+**
* **Ollama** installed and running
  👉 [Download here](https://ollama.ai/download)

---

### 🚀 Installation

1. Clone or download the repository

   ```bash
   git clone https://github.com/your-username/simple-local-rag.git
   cd simple-local-rag
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

---

### ▶️ Run It

```bash
python rag_ollama.py
```
---

## 🔐 Privacy & Local Processing

✅ 100% Local – No cloud uploads
✅ No API keys required
✅ All data stored localy
✅ Safe for confidential or private use

---

## 🪪 License

**MIT License** – free for personal and commercial use.

---

## ❤️ Credits

Built with:

* [Ollama](https://ollama.ai/) – local LLM inference
* [ChromaDB](https://www.trychroma.com/) – vector storage
* [SentenceTransformers](https://www.sbert.net/) – semantic embeddings
* [CrossEncoders](https://www.sbert.net/examples/applications/cross-encoder/) – reranking

---

⭐ **If you find this helpful, consider starring the project!**
