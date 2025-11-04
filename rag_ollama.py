"""
Enhanced Local RAG System with Advanced Features
- Intelligent chunking with overlap
- Hybrid retrieval (semantic + keyword)
- Query expansion and rewriting
- Context re-ranking
- Document metadata tracking
- Conversation history support
- Configurable retrieval strategies
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests

# ---------- CONFIGURATION ----------
@dataclass
class RAGConfig:
    embed_model_name: str = "all-MiniLM-L6-v2"
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gen_model: str = "llama3.2:3b"
    db_dir: str = "./enhanced_rag_db"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    max_conversation_history: int = 3

config = RAGConfig()

# ---------- LOAD MODELS ----------
print("🔄 Loading embedding model...")
embed_model = SentenceTransformer(config.embed_model_name)

print("🔄 Loading reranking model...")
rerank_model = CrossEncoder(config.rerank_model_name) if config.enable_reranking else None

print("✅ Models loaded successfully!")

# ---------- DOCUMENT PROCESSING ----------
class DocumentProcessor:
    """Enhanced document processing with intelligent chunking."""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Extract text from PDF with better formatting."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            return text
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def read_text(file_path: str) -> str:
        """Read text file with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        print(f"⚠️ Warning: Could not decode {file_path}, using ignore errors")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', '', text)
        return text.strip()
    
    @staticmethod
    def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create overlapping chunks with metadata."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': chunk_id,
                    'char_count': len(current_chunk)
                })
                
                # Create overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'char_count': len(current_chunk)
            })
        
        return chunks

# ---------- EMBEDDING & RETRIEVAL ----------
class EmbeddingManager:
    """Manages embeddings and vector operations."""
    
    @staticmethod
    def get_embedding(text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        if not text.strip():
            return None
        try:
            return embed_model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None
    
    @staticmethod
    def batch_embed(texts: List[str]) -> List[List[float]]:
        """Batch embedding for efficiency."""
        return embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

# ---------- QUERY ENHANCEMENT ----------
class QueryEnhancer:
    """Enhances queries through expansion and rewriting."""
    
    @staticmethod
    def expand_query(query: str, ollama_url: str = "http://localhost:11434/api/generate") -> List[str]:
        """Generate query variations using LLM."""
        if not config.enable_query_expansion:
            return [query]
        
        prompt = f"""Generate 2 alternative phrasings of this question that maintain the same meaning:

Original: {query}

Provide only the alternatives, one per line, without numbering or explanation."""
        
        try:
            response = requests.post(
                ollama_url,
                json={"model": config.gen_model, "prompt": prompt, "stream": False},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                alternatives = [q.strip() for q in result.split('\n') if q.strip()]
                return [query] + alternatives[:2]
        except:
            pass
        
        return [query]
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords

# ---------- RERANKING ----------
class Reranker:
    """Reranks retrieved documents using cross-encoder."""
    
    @staticmethod
    def rerank(query: str, documents: List[str], scores: List[float], top_k: int) -> Tuple[List[str], List[float]]:
        """Rerank documents using cross-encoder model."""
        if not config.enable_reranking or rerank_model is None:
            return documents[:top_k], scores[:top_k]
        
        try:
            pairs = [[query, doc] for doc in documents]
            rerank_scores = rerank_model.predict(pairs)
            
            # Combine with original scores
            combined_scores = [0.7 * rs + 0.3 * os for rs, os in zip(rerank_scores, scores)]
            
            # Sort by combined score
            ranked = sorted(zip(documents, combined_scores), key=lambda x: x[1], reverse=True)
            
            docs, scores = zip(*ranked[:top_k])
            return list(docs), list(scores)
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
            return documents[:top_k], scores[:top_k]

# ---------- DATABASE MANAGER ----------
class ChromaDBManager:
    """Manages ChromaDB operations with metadata."""
    
    def __init__(self):
        self.client = chromadb.Client(Settings(persist_directory=config.db_dir))
        self.collection = self.client.get_or_create_collection(
            "enhanced_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, file_path: str):
        """Add document with enhanced processing."""
        if not os.path.exists(file_path):
            print("❌ File not found.")
            return
        
        # Read document
        if file_path.endswith(".pdf"):
            text = DocumentProcessor.read_pdf(file_path)
        elif file_path.endswith(".txt"):
            text = DocumentProcessor.read_text(file_path)
        else:
            print(f"❌ Unsupported file type: {file_path}")
            return
        
        if not text.strip():
            print("❌ No text extracted from document.")
            return
        
        # Process and chunk
        text = DocumentProcessor.clean_text(text)
        chunks = DocumentProcessor.chunk_text_with_overlap(
            text, config.chunk_size, config.chunk_overlap
        )
        
        print(f"📚 Processing {len(chunks)} chunks from {os.path.basename(file_path)}...")
        
        # Batch embed
        chunk_texts = [c['text'] for c in chunks]
        embeddings = EmbeddingManager.batch_embed(chunk_texts)
        
        # Add to database
        valid_chunks = 0
        filename = os.path.basename(file_path)
        timestamp = datetime.now().isoformat()
        
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            try:
                self.collection.add(
                    ids=[f"{filename}_{timestamp}_{i}"],
                    documents=[chunk['text']],
                    embeddings=[emb],
                    metadatas=[{
                        'source': filename,
                        'chunk_id': chunk['chunk_id'],
                        'timestamp': timestamp,
                        'char_count': chunk['char_count']
                    }]
                )
                valid_chunks += 1
            except Exception as e:
                print(f"⚠️ Skipping chunk {i}: {e}")
        
        print(f"✅ Added {valid_chunks}/{len(chunks)} chunks to database.")
    
    def hybrid_search(self, query: str, query_variations: List[str]) -> Tuple[List[str], List[float]]:
        """Perform hybrid search with multiple query variations."""
        if self.collection.count() == 0:
            return [], []
        
        all_docs = []
        all_scores = []
        
        # Search with all query variations
        for q in query_variations:
            query_emb = EmbeddingManager.get_embedding(q)
            if query_emb is None:
                continue
            
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=config.top_k_retrieval
            )
            
            if results['documents'] and results['documents'][0]:
                all_docs.extend(results['documents'][0])
                # Convert distances to similarity scores
                distances = results['distances'][0]
                scores = [1 / (1 + d) for d in distances]
                all_scores.extend(scores)
        
        # Remove duplicates while keeping best scores
        doc_scores = {}
        for doc, score in zip(all_docs, all_scores):
            if doc not in doc_scores or score > doc_scores[doc]:
                doc_scores[doc] = score
        
        # Sort by score
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        docs, scores = zip(*sorted_items) if sorted_items else ([], [])
        
        return list(docs), list(scores)
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection.name
        }

# ---------- LLM INTERACTION ----------
class LLMGenerator:
    """Handles LLM generation with conversation context."""
    
    def __init__(self):
        self.conversation_history = []
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from Ollama."""
        url = "http://localhost:11434/api/generate"
        data = {"model": config.gen_model, "prompt": prompt}
        
        try:
            response = requests.post(url, json=data, stream=True, timeout=180)
            response.raise_for_status()
            
            final_text = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        final_text += obj["response"]
                except json.JSONDecodeError:
                    continue
            
            return final_text.strip()
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return "Error generating response."
    
    def add_to_history(self, question: str, answer: str):
        """Add exchange to conversation history."""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.conversation_history) > config.max_conversation_history:
            self.conversation_history.pop(0)
    
    def get_context_prompt(self, question: str, context: str) -> str:
        """Build prompt with conversation history."""
        history_text = ""
        if self.conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for h in self.conversation_history[-2:]:
                history_text += f"Q: {h['question']}\nA: {h['answer'][:200]}...\n"
        
        prompt = f"""Use the provided context to answer the question accurately and concisely.{history_text}

Context:
{context}

Question: {question}

Answer (be specific and cite relevant details from the context):"""
        
        return prompt

# ---------- MAIN RAG SYSTEM ----------
class EnhancedRAG:
    """Main RAG system orchestrator."""
    
    def __init__(self):
        self.db_manager = ChromaDBManager()
        self.llm = LLMGenerator()
        self.query_enhancer = QueryEnhancer()
        self.reranker = Reranker()
    
    def add_document(self, file_path: str):
        """Add document to RAG system."""
        self.db_manager.add_document(file_path)
    
    def query(self, question: str, show_sources: bool = False):
        """Query the RAG system with enhanced retrieval."""
        if self.db_manager.collection.count() == 0:
            print("⚠️ Database is empty! Add documents first.")
            return
        
        print("\n🔍 Enhancing query...")
        query_variations = self.query_enhancer.expand_query(question)
        
        print(f"📊 Searching with {len(query_variations)} query variation(s)...")
        docs, scores = self.db_manager.hybrid_search(question, query_variations)
        
        if not docs:
            print("⚠️ No relevant documents found.")
            return
        
        print(f"🎯 Reranking top {len(docs)} results...")
        docs, scores = self.reranker.rerank(question, docs, scores, config.top_k_rerank)
        
        # Build context
        context = "\n\n---\n\n".join(docs)
        
        if show_sources:
            print(f"\n📚 Using {len(docs)} context chunks (scores: {[f'{s:.3f}' for s in scores]})")
        
        print("\n🧠 Generating answer...\n")
        prompt = self.llm.get_context_prompt(question, context)
        answer = self.llm.generate_response(prompt)
        
        print("=" * 80)
        print("🧠 ANSWER:\n")
        print(answer)
        print("=" * 80)
        
        self.llm.add_to_history(question, answer)
    
    def show_stats(self):
        """Display system statistics."""
        stats = self.db_manager.get_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Conversation history: {len(self.llm.conversation_history)} exchanges")

# ---------- MAIN INTERFACE ----------
def main():
    print("\n" + "=" * 80)
    print("🚀 ENHANCED LOCAL RAG SYSTEM")
    print("=" * 80)
    print(f"📦 Embeddings: {config.embed_model_name}")
    print(f"🤖 LLM: {config.gen_model}")
    print(f"🔧 Reranking: {'Enabled' if config.enable_reranking else 'Disabled'}")
    print(f"🔧 Query Expansion: {'Enabled' if config.enable_query_expansion else 'Disabled'}")
    print("=" * 80)
    
    rag = EnhancedRAG()
    
    while True:
        print("\n📋 OPTIONS:")
        print("1️⃣  Add document (.txt or .pdf)")
        print("2️⃣  Ask a question")
        print("3️⃣  Show statistics")
        print("4️⃣  Clear conversation history")
        print("5️⃣  Exit")
        
        choice = input("\n👉 Choose (1-5): ").strip()
        
        if choice == "1":
            path = input("📁 Enter file path: ").strip().strip('"')
            rag.add_document(os.path.normpath(path))
        
        elif choice == "2":
            question = input("❓ Your question: ").strip()
            if question:
                show_sources = input("Show source scores? (y/n): ").lower() == 'y'
                rag.query(question, show_sources)
        
        elif choice == "3":
            rag.show_stats()
        
        elif choice == "4":
            rag.llm.conversation_history.clear()
            print("✅ Conversation history cleared.")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
