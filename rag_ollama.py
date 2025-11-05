import os
import re
import json
import hashlib
import csv
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import time

# Core imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import requests
    import numpy as np
    from tqdm import tqdm
    from PyPDF2 import PdfReader
except ImportError as e:
    print("❌ Missing required package. Please run:")
    print("   pip install chromadb sentence-transformers requests numpy tqdm PyPDF2")
    exit(1)

# Optional imports for extra formats
try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except:
    DOCX_SUPPORT = False

try:
    from pptx import Presentation
    PPTX_SUPPORT = True
except:
    PPTX_SUPPORT = False

try:
    import openpyxl
    EXCEL_SUPPORT = True
except:
    EXCEL_SUPPORT = False

try:
    from bs4 import BeautifulSoup
    HTML_SUPPORT = True
except:
    HTML_SUPPORT = False

# ---------- SIMPLE CONFIGURATION ----------
class Config:
    """Simple configuration with smart defaults."""
    
    def __init__(self):
        # Paths
        self.data_folder = "./my_knowledge_base"
        self.cache_folder = "./cache"
        
        # Smart defaults
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"
        
        # Auto-detected on first run
        self.embedding_model = None
        self.use_reranking = True
        self.language = "auto"
        
        # User preferences
        self.show_sources = True
        self.stream_answers = True
        self.save_history = True
        
    def load(self):
        """Load saved settings."""
        config_file = os.path.join(self.data_folder, "settings.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    setattr(self, key, value)
    
    def save(self):
        """Save settings."""
        os.makedirs(self.data_folder, exist_ok=True)
        config_file = os.path.join(self.data_folder, "settings.json")
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

config = Config()

# ---------- FIRST TIME SETUP WIZARD ----------
def setup_wizard():
    """Friendly setup wizard for first-time users."""
    print("\n" + "="*80)
    print("👋 WELCOME TO YOUR PERSONAL KNOWLEDGE ASSISTANT!")
    print("="*80)
    print("\nLet's set things up quickly (takes 1 minute)...\n")
    
    # 1. Check Ollama
    print("1️⃣ Checking if Ollama is running...")
    try:
        response = requests.get(f"{config.ollama_url}/api/tags", timeout=3)
        if response.status_code == 200:
            print("   ✅ Ollama is running!")
            models = response.json().get('models', [])
            if models:
                print(f"   📦 Available models: {', '.join([m['name'] for m in models[:3]])}")
        else:
            print("   ⚠️ Ollama not responding. Make sure it's running.")
            print("   💡 Visit: https://ollama.ai/download")
    except:
        print("   ⚠️ Can't connect to Ollama. Make sure it's running.")
        print("   💡 Visit: https://ollama.ai/download")
        input("\n   Press Enter when Ollama is ready...")
    
    # 2. Language preference
    print("\n2️⃣ What language will you use most?")
    print("   1. English")
    print("   2. French")
    print("   3. Spanish")
    print("   4. Arabic")
    print("   5. Multiple languages")
    print("   6. Other")
    
    choice = input("\n   Choose (1-6) or press Enter for English: ").strip() or "1"
    
    language_map = {
        "1": ("english", "all-MiniLM-L6-v2"),
        "2": ("french", "paraphrase-multilingual-MiniLM-L12-v2"),
        "3": ("spanish", "paraphrase-multilingual-MiniLM-L12-v2"),
        "4": ("arabic", "paraphrase-multilingual-MiniLM-L12-v2"),
        "5": ("multilingual", "paraphrase-multilingual-MiniLM-L12-v2"),
        "6": ("multilingual", "paraphrase-multilingual-MiniLM-L12-v2")
    }
    
    config.language, config.embedding_model = language_map.get(choice, ("english", "all-MiniLM-L6-v2"))
    print(f"   ✅ Set to: {config.language}")
    
    # 3. Performance vs Quality
    print("\n3️⃣ Do you prefer:")
    print("   1. Faster answers (good quality)")
    print("   2. Best quality answers (slightly slower)")
    
    perf = input("\n   Choose (1-2) or press Enter for faster: ").strip() or "1"
    config.use_reranking = (perf == "2")
    print(f"   ✅ Set to: {'Best quality' if config.use_reranking else 'Faster answers'}")
    
    # 4. Save settings
    config.save()
    
    print("\n" + "="*80)
    print("✅ Setup complete! Let's load your AI assistant...")
    print("="*80)
    time.sleep(2)

# ---------- SMART DOCUMENT READER ----------
class SmartReader:
    """Reads any document automatically."""
    
    @staticmethod
    def read(file_path: str) -> str:
        """Read any file and extract text."""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.pdf':
                return SmartReader._read_pdf(file_path)
            elif ext in ['.docx', '.doc'] and DOCX_SUPPORT:
                return SmartReader._read_docx(file_path)
            elif ext in ['.pptx', '.ppt'] and PPTX_SUPPORT:
                return SmartReader._read_pptx(file_path)
            elif ext in ['.xlsx', '.xls'] and EXCEL_SUPPORT:
                return SmartReader._read_excel(file_path)
            elif ext == '.csv':
                return SmartReader._read_csv(file_path)
            elif ext in ['.html', '.htm'] and HTML_SUPPORT:
                return SmartReader._read_html(file_path)
            else:
                return SmartReader._read_text(file_path)
        except Exception as e:
            print(f"   ⚠️ Couldn't read {os.path.basename(file_path)}: {e}")
            return ""
    
    @staticmethod
    def _read_text(path):
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except:
                continue
        return ""
    
    @staticmethod
    def _read_pdf(path):
        try:
            reader = PdfReader(path)
            return "\n\n".join([p.extract_text() or "" for p in reader.pages])
        except:
            return ""
    
    @staticmethod
    def _read_docx(path):
        doc = DocxDocument(path)
        return '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    
    @staticmethod
    def _read_pptx(path):
        prs = Presentation(path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return '\n\n'.join(text)
    
    @staticmethod
    def _read_excel(path):
        wb = openpyxl.load_workbook(path, data_only=True)
        text = []
        for sheet in wb.sheetnames:
            for row in wb[sheet].iter_rows(values_only=True):
                text.append(' | '.join([str(c) for c in row if c]))
        return '\n'.join(text)
    
    @staticmethod
    def _read_csv(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return '\n'.join([' | '.join(row) for row in csv.reader(f)])
    
    @staticmethod
    def _read_html(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator='\n')

# ---------- EASY RAG SYSTEM ----------
class EasyRAG:
    """Simple RAG system that just works."""
    
    def __init__(self):
        print("\n🔄 Loading your AI assistant...")
        
        # Setup folders
        os.makedirs(config.data_folder, exist_ok=True)
        os.makedirs(config.cache_folder, exist_ok=True)
        
        # Load models
        print("   📥 Loading AI models (first time may take a minute)...")
        self.embed_model = SentenceTransformer(config.embedding_model)
        
        if config.use_reranking:
            self.rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            self.rerank_model = None
        
        # Setup database
        self.db = chromadb.PersistentClient(path=config.data_folder)
        self.collection = self.db.get_or_create_collection("documents")
        
        # Load cache
        self.cache_file = os.path.join(config.cache_folder, "answers.pkl")
        self.cache = self._load_cache()
        
        # Load document registry
        self.docs_file = os.path.join(config.data_folder, "documents.json")
        self.docs = self._load_docs()
        
        # Conversation history
        self.history = []
        
        print("   ✅ Ready to help!\n")
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _load_docs(self):
        if os.path.exists(self.docs_file):
            with open(self.docs_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_docs(self):
        with open(self.docs_file, 'w') as f:
            json.dump(self.docs, f, indent=2)
    
    def add_document(self, file_path: str):
        """Add a document to your knowledge base."""
        filename = os.path.basename(file_path)
        
        # Check if already added
        if filename in self.docs:
            print(f"   ℹ️ {filename} is already in your knowledge base")
            return
        
        print(f"   📖 Reading {filename}...")
        text = SmartReader.read(file_path)
        
        if not text.strip():
            print(f"   ⚠️ No text found in {filename}")
            return
        
        # Split into chunks
        print(f"   ✂️ Processing...")
        chunks = self._smart_chunk(text)
        
        # Create embeddings
        print(f"   🧠 Learning from {len(chunks)} sections...")
        texts = [c['text'] for c in chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()
        
        # Add to database
        timestamp = datetime.now().isoformat()
        ids = [f"{filename}_{timestamp}_{i}" for i in range(len(chunks))]
        metadatas = [{'source': filename, 'added': timestamp} for _ in chunks]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # Update registry
        self.docs[filename] = {
            'path': file_path,
            'chunks': len(chunks),
            'added': timestamp,
            'size': os.path.getsize(file_path)
        }
        self._save_docs()
        
        print(f"   ✅ Added {filename} to your knowledge base!")
    
    def add_folder(self, folder_path: str):
        """Add all documents from a folder."""
        path = Path(folder_path)
        
        if not path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Find all readable files
        extensions = ['.pdf', '.txt', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.html', '.md']
        files = [f for f in path.rglob('*') if f.suffix.lower() in extensions]
        
        if not files:
            print("⚠️ No documents found in this folder")
            return
        
        print(f"\n📁 Found {len(files)} documents. Adding them...")
        
        for file in files:
            try:
                self.add_document(str(file))
            except Exception as e:
                print(f"   ⚠️ Skipped {file.name}: {e}")
        
        print(f"\n✅ Done! Your knowledge base now has {len(self.docs)} documents")
    
    def _smart_chunk(self, text: str, size: int = 600) -> List[Dict]:
        """Split text into smart chunks."""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) > size and current:
                chunks.append({'text': current.strip()})
                # Keep some overlap
                words = current.split()
                current = ' '.join(words[-20:]) + ' ' + sent
            else:
                current += ' ' + sent
        
        if current.strip():
            chunks.append({'text': current.strip()})
        
        return chunks
    
    def ask(self, question: str, use_documents: bool = True) -> str:
        """Ask a question and get an answer."""
        
        # If user doesn't want to use documents (general knowledge questions)
        if not use_documents:
            print("   💭 Using general knowledge...")
            prompt = f"Answer this question concisely and accurately:\n\n{question}"
            return self._generate(prompt)
        
        # Check cache first
        question_key = question.lower().strip()
        if question_key in self.cache:
            cached = self.cache[question_key]
            age = datetime.now() - datetime.fromisoformat(cached['time'])
            if age < timedelta(hours=24):
                print("\n" + "="*80)
                print("📝 ANSWER (from memory):\n")
                print(cached['answer'])
                print("="*80)
                if config.show_sources:
                    print(f"\n💡 Sources: {', '.join(cached['sources'])}")
                return cached['answer']
        
        # Check if knowledge base is empty
        if self.collection.count() == 0:
            return "I don't have any documents to search yet. Please add some documents first, or ask me to use my general knowledge by saying 'from your knowledge' or 'without documents'."
        
        # Search for relevant information
        print("   🔍 Searching your documents...")
        
        query_embedding = self.embed_model.encode(question).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'][0]:
            return "I couldn't find any information about that in your documents. Try asking something else or add more documents."
        
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        # Rerank if enabled
        if self.rerank_model:
            print("   🎯 Finding the best matches...")
            pairs = [[question, doc] for doc in docs]
            scores = self.rerank_model.predict(pairs)
            sorted_indices = np.argsort(scores)[::-1]
            docs = [docs[i] for i in sorted_indices[:5]]
            metas = [metas[i] for i in sorted_indices[:5]]
        else:
            docs = docs[:5]
            metas = metas[:5]
        
        # Build context
        context = "\n\n".join(docs)
        sources = list(set([m['source'] for m in metas]))
        
        # Generate answer
        print("   💭 Thinking...")
        
        prompt = f"""Based on the following information, answer the question accurately and concisely.

Information:
{context}

Question: {question}

Answer (be specific and helpful):"""
        
        answer = self._generate(prompt)
        
        # Cache the answer
        self.cache[question_key] = {
            'answer': answer,
            'sources': sources,
            'time': datetime.now().isoformat()
        }
        self._save_cache()
        
        # Show sources
        if config.show_sources:
            print(f"\n💡 Sources: {', '.join(sources)}")
        
        return answer
    
    def _generate(self, prompt: str) -> str:
        """Generate answer from LLM."""
        url = f"{config.ollama_url}/api/generate"
        
        payload = {
            "model": config.model_name,
            "prompt": prompt,
            "stream": config.stream_answers,
            "options": {"temperature": 0.1}
        }
        
        try:
            response = requests.post(url, json=payload, stream=config.stream_answers, timeout=120)
            
            if not config.stream_answers:
                return response.json().get('response', 'Error generating answer')
            
            print("\n" + "="*80)
            print("📝 ANSWER:\n")
            
            full_answer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    try:
                        obj = json.loads(line)
                        chunk = obj.get('response', '')
                        full_answer += chunk
                        print(chunk, end='', flush=True)
                    except:
                        continue
            
            print("\n" + "="*80)
            return full_answer
            
        except Exception as e:
            return f"Error: Couldn't connect to Ollama. Make sure it's running. ({e})"
    
    def list_documents(self):
        """Show all documents in knowledge base."""
        if not self.docs:
            print("\n📭 Your knowledge base is empty.")
            print("💡 Add documents with: Add Documents")
            return
        
        print("\n" + "="*80)
        print("📚 YOUR KNOWLEDGE BASE")
        print("="*80)
        
        for i, (filename, info) in enumerate(self.docs.items(), 1):
            size = info['size'] / 1024  # KB
            added = datetime.fromisoformat(info['added']).strftime('%Y-%m-%d %H:%M')
            print(f"\n{i}. {filename}")
            print(f"   📊 {info['chunks']} sections | 💾 {size:.1f} KB")
            print(f"   🕒 Added: {added}")
        
        print("\n" + "="*80)
        print(f"Total: {len(self.docs)} documents, {self.collection.count()} sections")
    
    def remove_document(self, filename: str):
        """Remove a document from knowledge base."""
        if filename not in self.docs:
            print(f"❌ {filename} not found")
            return
        
        # Remove from database
        results = self.collection.get(where={"source": filename})
        if results['ids']:
            self.collection.delete(ids=results['ids'])
        
        # Remove from registry
        del self.docs[filename]
        self._save_docs()
        
        print(f"✅ Removed {filename}")
    
    def chat(self):
        """Interactive chat mode with smart routing."""
        print("\n💬 CHAT MODE")
        print("="*80)
        print("Ask me anything! I'll search your documents or use general knowledge.")
        print("\n💡 Tips:")
        print("   • Regular questions: I'll search your documents")
        print("   • Say 'from your knowledge' or 'without documents': I'll use general AI knowledge")
        print("   • Just chat naturally - say hi, thanks, etc!")
        print("\n📋 Commands:")
        print("   • 'exit' or 'quit' - Leave chat mode")
        print("   • 'docs' - See your documents")
        print("   • 'clear' - Reset conversation")
        print("="*80 + "\n")
        
        # Casual conversation patterns
        casual_patterns = [
            r'^(hi|hello|hey|bonjour|salut)',
            r'^(thanks|thank you|merci)',
            r'^(bye|goodbye|au revoir)',
            r'^(ok|okay|good|nice|great|cool|awesome)',
            r'(nice work|well done|good job)',
            r'^(yes|no|oui|non)'
        ]
        
        # General knowledge indicators
        general_knowledge_patterns = [
            r'(from your|use your|with your).*(knowledge|brain|memory)',
            r'(without|don\'t use|ignore).*(document|file|pdf)',
            r'(general|common|basic).*(knowledge|information)',
            r'answer.*your.*knowledge',
        ]
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                # Check for exit commands
                if question.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\n👋 Leaving chat mode...")
                    break
                
                # Check for special commands
                if question.lower() == 'docs':
                    self.list_documents()
                    continue
                
                if question.lower() in ['clear', 'reset']:
                    self.history.clear()
                    print("✅ Chat history cleared\n")
                    continue
                
                # Detect casual conversation
                is_casual = any(re.search(pattern, question.lower()) for pattern in casual_patterns)
                
                if is_casual:
                    # Respond casually without searching documents
                    casual_responses = {
                        'hi': "Hello! How can I help you today? Ask me anything about your documents!",
                        'hello': "Hi there! What would you like to know?",
                        'hey': "Hey! What can I help you with?",
                        'bonjour': "Bonjour! Comment puis-je vous aider?",
                        'thanks': "You're welcome! Happy to help! 😊",
                        'thank you': "My pleasure! Let me know if you need anything else!",
                        'merci': "De rien! N'hésitez pas si vous avez d'autres questions!",
                        'bye': "Goodbye! Your knowledge base is always here when you need it!",
                        'goodbye': "See you later! Feel free to come back anytime!",
                        'ok': "Great! What else would you like to know?",
                        'nice': "Thank you! I'm here to help! 😊",
                        'cool': "Glad you like it! What else can I help with?",
                        'good': "Excellent! Anything else you'd like to know?",
                    }
                    
                    # Find matching response
                    response = None
                    for key, resp in casual_responses.items():
                        if key in question.lower():
                            response = resp
                            break
                    
                    if not response:
                        response = "I'm here to help! Ask me anything about your documents, or just chat!"
                    
                    print(f"\n{response}\n")
                    continue
                
                # Detect general knowledge request
                use_general_knowledge = any(re.search(pattern, question.lower()) for pattern in general_knowledge_patterns)
                
                if use_general_knowledge:
                    # Clean the question (remove the "from your knowledge" part)
                    clean_question = question
                    for pattern in general_knowledge_patterns:
                        clean_question = re.sub(pattern, '', clean_question, flags=re.IGNORECASE)
                    clean_question = clean_question.strip(' ,-')
                    
                    if not clean_question:
                        clean_question = question
                    
                    print()
                    answer = self.ask(clean_question, use_documents=False)
                    print()
                else:
                    # Normal document-based question
                    print()
                    answer = self.ask(question, use_documents=True)
                    print()
                
                # Save to history
                self.history.append({
                    'question': question,
                    'answer': answer,
                    'time': datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Leaving chat mode...")
                break
            except Exception as e:
                print(f"\n❌ Oops! Something went wrong: {e}")
                print("💡 Try asking in a different way, or type 'exit' to leave chat mode\n")

# ---------- SIMPLE MENU ----------
def main():
    """Simple, friendly main menu."""
    
    # First time setup
    if not os.path.exists(os.path.join(config.data_folder, "settings.json")):
        setup_wizard()
    else:
        config.load()
    
    # Initialize system
    rag = EasyRAG()
    
    # Main loop
    while True:
        print("\n" + "="*80)
        print("🏠 MAIN MENU")
        print("="*80)
        print("\n1. 💬 Chat with your documents")
        print("2. ➕ Add documents")
        print("3. 📁 Add entire folder")
        print("4. 📚 View my documents")
        print("5. 🗑️  Remove a document")
        print("6. ⚙️  Settings")
        print("7. ❓ Help")
        print("8. 🚪 Exit")
        
        choice = input("\n👉 Choose (1-8): ").strip()
        
        if choice == '1':
            # Chat mode
            if rag.collection.count() == 0:
                print("\n⚠️ Your knowledge base is empty!")
                print("💡 Add some documents first (option 2 or 3)")
                continue
            rag.chat()
        
        elif choice == '2':
            # Add single documents
            print("\n📂 Drag and drop files here, or type the path")
            print("💡 Supported: PDF, Word, Excel, PowerPoint, Text, CSV, HTML")
            print("💡 Separate multiple files with semicolon (;)")
            
            paths = input("\n📄 File path(s): ").strip().strip('"')
            
            if not paths:
                continue
            
            for path in paths.split(';'):
                path = path.strip().strip('"')
                if os.path.exists(path):
                    try:
                        rag.add_document(path)
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                else:
                    print(f"   ❌ File not found: {path}")
        
        elif choice == '3':
            # Add folder
            print("\n📁 This will add all documents from a folder")
            path = input("📂 Folder path: ").strip().strip('"')
            
            if os.path.exists(path):
                rag.add_folder(path)
            else:
                print("❌ Folder not found")
        
        elif choice == '4':
            # List documents
            rag.list_documents()
        
        elif choice == '5':
            # Remove document
            rag.list_documents()
            if rag.docs:
                filename = input("\n🗑️ Enter filename to remove: ").strip()
                rag.remove_document(filename)
        
        elif choice == '6':
            # Settings
            print("\n" + "="*80)
            print("⚙️ SETTINGS")
            print("="*80)
            print(f"\n1. Show sources: {'✓' if config.show_sources else '✗'}")
            print(f"2. Stream answers: {'✓' if config.stream_answers else '✗'}")
            print(f"3. Language: {config.language}")
            print(f"4. Quality mode: {'Best quality' if config.use_reranking else 'Faster'}")
            print(f"5. Ollama model: {config.model_name}")
            print("6. Back")
            
            setting = input("\n👉 Choose (1-6): ").strip()
            
            if setting == '1':
                config.show_sources = not config.show_sources
                print(f"✅ Show sources: {'ON' if config.show_sources else 'OFF'}")
                config.save()
            elif setting == '2':
                config.stream_answers = not config.stream_answers
                print(f"✅ Stream answers: {'ON' if config.stream_answers else 'OFF'}")
                config.save()
            elif setting == '3':
                print("\n1. English")
                print("2. French")
                print("3. Spanish")
                print("4. Multiple languages")
                lang = input("Choose: ").strip()
                if lang in ['1', '2', '3', '4']:
                    print("✅ Language updated. Restart to apply changes.")
            elif setting == '4':
                config.use_reranking = not config.use_reranking
                print(f"✅ Quality: {'Best' if config.use_reranking else 'Faster'}")
                print("   ℹ️ Restart to apply changes")
                config.save()
            elif setting == '5':
                new_model = input(f"Model name (current: {config.model_name}): ").strip()
                if new_model:
                    config.model_name = new_model
                    config.save()
                    print("✅ Model updated")
        
        elif choice == '7':
            # Help
            print("\n" + "="*80)
            print("❓ QUICK HELP")
            print("="*80)
            print("\n📖 How to use:")
            print("   1. Add your documents (PDF, Word, etc.)")
            print("   2. Ask questions in chat mode")
            print("   3. Get instant answers from your documents!")
            print("\n💡 Tips:")
            print("   • Be specific in your questions")
            print("   • Add more documents for better answers")
            print("   • Use 'docs' command in chat to see your documents")
            print("\n🔧 Troubleshooting:")
            print("   • If answers are slow: Turn off 'Best quality' mode")
            print("   • If no answers: Make sure documents contain relevant info")
            print("   • If Ollama error: Make sure Ollama is running")
            print("\n📚 Learn more: github.com/ollama/ollama")
        
        elif choice == '8':
            print("\n👋 Goodbye! Your knowledge base is saved.")
            print(f"📁 Location: {config.data_folder}")
            break
        
        else:
            print("❌ Invalid choice. Please choose 1-8")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try restarting the program")
