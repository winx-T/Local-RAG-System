# 🧠 Local RAG System – Smart, Simple, and Powerful

A **zero-setup**, **fully local Retrieval-Augmented Generation (RAG) system** that lets you instantly ask questions about your own **PDFs, Word, Excel, PowerPoint, Text, CSV, or HTML files** — **with no cloud, no API keys, and complete privacy**.

Built for **everyone**:
- 🧩 Beginners who just want a simple, working personal AI.
- 🧠 Power users who want reranking, hybrid search, and multilingual embeddings.

##  What’s New in this version
| Feature                        | Description                                                               |
| ------------------------------ | ------------------------------------------------------------------------- |
| 💡 **General Knowledge Mode**  | Ask questions without documents (“from your knowledge”).                  |
| 💬 **Smarter Chat System**     | Handles casual talk, document Q&A, or general queries automatically.      |
| 🧠 **Intent Detection**        | Detects whether to use documents or general knowledge via regex patterns. |
| 🧰 **Improved Cache Handling** | Displays “from memory” responses for cached answers.                      |
| 🛡️ **Error Resilience**       | Handles runtime issues gracefully without crashes.                        |
| ✨ **Enhanced UX**              | Clearer chat messages, emojis, and helpful prompts.                       |


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

## 🚀 Highlights

| Feature | Description |
|----------|--------------|
| ⚙️ **Zero Configuration** | Works out of the box – just run the script |
| 🧭 **Setup Wizard** | Friendly first-run setup: language, speed, and quality |
| 💬 **Plain English Interface** | No technical jargon – “documents,” not “embeddings” |
| 🧠 **Smart Chunking** | Sentence-aware text splitting with overlap |
| 🎯 **Cross-Encoder Re-ranking** | High-accuracy context selection using `ms-marco-MiniLM-L-6-v2` |
| 🔍 **Semantic Search** | Embeddings via `sentence-transformers` |
| 🕰️ **Conversation Memory** | Keeps track of your recent Q&A exchanges |
| 💾 **Persistent Knowledge Base** | Stores your documents locally using **ChromaDB** |
| 🛡️ **100% Local Privacy** | All processing stays on your machine – no cloud calls |
| 🌍 **Multilingual Support** | English, French, Arabic, Spanish, and more |

---

## 🧩 Core Components

### 🧭 1️⃣ First-Time Setup Wizard
On first run, you’ll be guided through:
1. Checking if Ollama is running  
2. Choosing your language (English, French, Arabic, etc.)  
3. Selecting **speed** or **best quality** mode  

Your preferences are saved automatically for future sessions.

---

### 📄 2️⃣ Smart Document Reader
Automatically extracts text from:
- PDF (.pdf)
- Word (.docx, .doc)
- Excel (.xlsx, .xls)
- PowerPoint (.pptx, .ppt)
- Text & Markdown (.txt, .md)
- CSV (.csv)
- HTML (.html, .htm)

🧠 It also:
- Auto-detects formats  
- Skips unreadable files gracefully  
- Uses multiple encodings for compatibility  

---

### ✂️ 3️⃣ Intelligent Chunking
Splits text into **~600-character segments** with a **20-word overlap**  
→ Ensures smooth transitions between chunks and preserves sentence meaning.

---

### 📚 4️⃣ Local Knowledge Base
Documents are:
- Embedded using **SentenceTransformers**
- Stored persistently with **ChromaDB**
- Tagged with metadata (source, size, timestamp)

---

### 🔎 5️⃣ Semantic + Reranked Search
1. Embeds your question into vector space  
2. Retrieves top 20 relevant sections  
3. (Optional) Reranks them using **CrossEncoder**  
4. Assembles the top chunks as context for the LLM

💡 Results are cached for 24 hours to speed up repeated queries.

---

### 💬 6️⃣ Chat Mode
Interactive chat with your documents:

````bash
You: What is cloud computing?
🔍 Searching your documents...
💭 Thinking...

📝 ANSWER:
Cloud computing is a model for delivering computing services over the internet...

💡 Sources: cloud_intro.pdf
`````

🧰 Commands:

* `docs` → show your document list
* `clear` → reset chat memory
* `exit` → quit chat mode

---

## ⚙️ Installation

### 🧠 Requirements

* **Python 3.8+**
* **Ollama** installed and running
  👉 [Download here](https://ollama.ai/download)

---

### 📦 Install Dependencies

```bash
pip install chromadb sentence-transformers requests numpy tqdm PyPDF2
# Optional (for extra formats)
pip install python-docx python-pptx openpyxl beautifulsoup4
```

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

**First run** → setup wizard (language + model preferences)
**Next runs** → jump straight into chat or add new docs

---

## 🏠 Main Menu

```
🏠 MAIN MENU
1. 💬 Chat with your documents
2. ➕ Add documents
3. 📁 Add entire folder
4. 📚 View my documents
5. 🗑️ Remove a document
6. ⚙️ Settings
7. ❓ Help
8. 🚪 Exit
```

---

## ⚙️ Settings Menu

| Option         | Description                                |
| -------------- | ------------------------------------------ |
| Show sources   | Toggle document sources in answers         |
| Stream answers | Stream text as it’s generated              |
| Language       | Change preferred language                  |
| Quality mode   | Switch between faster or best-quality mode |
| Ollama model   | Set LLM (e.g., `llama3.2:3b`)              |

---

## 🧠 Example Workflow

```bash
python rag_ollama.py
```

```
👉 Choose (1-8): 2
📂 Add your documents

👉 Choose (1-8): 1
💬 Ask: What is machine learning?

🧠 ANSWER:
Machine learning is a subset of AI that enables systems to learn from data...
💡 Sources: ai_intro.pdf
```

---

## 📊 Performance Comparison

| Mode    | Description    | Retrieval Accuracy | Speed                             |
| ------- | -------------- | ------------------ | --------------------------------- |
| ⚡ Fast  | No reranking   | 70%                | Very fast                         |
| 🎯 Best | With reranking | 87%                | Slightly slower but more accurate |

---

## 🔐 Privacy & Local Processing

✅ 100% Local – No cloud uploads
✅ No API keys required
✅ All data stored under `./my_knowledge_base`
✅ Safe for confidential or private use

---

## 🧰 Troubleshooting

| Problem              | Solution                                    |
| -------------------- | ------------------------------------------- |
| ❌ Ollama not found   | Run `ollama serve`                          |
| ⚠️ Slow response     | Disable reranking (Settings → Quality Mode) |
| 💾 High memory usage | Reduce chunk size or disable reranking      |
| 📁 No answers        | Add more relevant documents                 |
| 🔌 Connection error  | Ensure Ollama is running locally            |

---

## 🚀 Future Roadmap

* [ ] Web-based UI
* [ ] Multi-modal retrieval (images, tables)
* [ ] Document comparison & citation linking
* [ ] Graph-based search
* [ ] Fine-tuning assistant behavior

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
