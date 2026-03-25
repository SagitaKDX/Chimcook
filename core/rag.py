import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import warnings
# Suppress noisy warnings from huggingface
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except ImportError:
    raise ImportError(
        "Missing RAG dependencies. Please install: "
        "pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu"
    )

class RAGPipeline:
    """
    Offline Retrieval-Augmented Generation (RAG) Module for Voice Assistant.
    
    Optimized for:
    - CPU-only execution (Intel Core i7)
    - Low RAM usage (< 8GB system limit)
    - Zero cold-start latency (pre-loaded models)
    """

    def __init__(
        self,
        docs_dir: str = "data/docs",
        faiss_index_path: str = "data/faiss_index",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3.2:1b", # Often exact tag in Ollama
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initializes models into memory to ensure zero cold-start latency.
        """
        self.docs_dir = Path(docs_dir)
        self.faiss_index_path = Path(faiss_index_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print("[RAG] Initializing RAG Pipeline (Offline Mode)...")

        # 1. Load Local Embedding Model (Fast, small memory footprint ~90MB)
        # We use CPU explicitly.
        print(f"[RAG] Loading Embedding Model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. Load or Create FAISS Index
        self.vector_store = None
        self._load_or_create_index()

        # 3. Initialize Local LLM connection
        # (Optional) Connect to a locally running Ollama instance pre-loaded with the model
        self.llm = None
        try:
            print(f"[RAG] Connecting to Ollama LLM Engine: {ollama_model} (Optional)")
            self.llm = Ollama(
                model=ollama_model,
                num_ctx=2048,
                num_thread=4,
                temperature=0.3
            )
        except Exception as e:
            print(f"[RAG] Ollama not available ({e}). RAG will run in context-only mode.")

        # 4. Define strict RAG prompt
        self.prompt_template = PromptTemplate(
            template=(
                "You are an offline conversational AI assistant. Use ONLY the following context to answer the user's question.\n"
                "If the context does not contain the answer, say 'I cannot answer that based on my knowledge base.' Do NOT hallucinate.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer concisely:"
            ),
            input_variables=["context", "question"]
        )
        print("[RAG] Initialization Complete. System Ready.")

    def _load_or_create_index(self):
        """Loads FAISS index from disk if it exists, otherwise prepares an empty state."""
        if self.faiss_index_path.exists():
            print(f"[RAG] Loading existing FAISS index from {self.faiss_index_path}...")
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.faiss_index_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True # Required for local trusts
                )
                print("[RAG] FAISS index loaded successfully.")
            except Exception as e:
                print(f"[RAG] Failed to load index: {e}. Starting fresh.")
                self.vector_store = None
        else:
            print("[RAG] No existing FAISS index found. Ready to ingest documents.")

    def ingest_documents(self):
        """
        Reads all .md and .txt files from docs_dir, chunks them, and builds the FAISS index.
        Runs once usually during setup/configuration.
        """
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RAG] Created empty docs directory: {self.docs_dir}. Place .md/.txt files here.")
            return

        file_paths = []
        file_paths.extend(self.docs_dir.glob("**/*.md"))
        file_paths.extend(self.docs_dir.glob("**/*.txt"))

        if not file_paths:
            print(f"[RAG] No documents found in {self.docs_dir} to ingest.")
            return

        documents = []
        print(f"[RAG] Found {len(file_paths)} documents. Ingesting...")
        
        # Load documents
        for fp in file_paths:
            try:
                loader = TextLoader(str(fp), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                print(f"[RAG] Error loading {fp}: {e}")

        # Chunking: Recursive splitting to respect semantic boundaries (paragraphs/sentences)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"[RAG] Split into {len(chunks)} chunks. Building FAISS index...")

        # Build and save FAISS index
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        # Persist to disk
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.faiss_index_path))
        print(f"[RAG] FAISS index updated and saved to {self.faiss_index_path}.")

    def get_context(self, user_query: str, top_k: int = 2) -> str:
        """Lightweight method to just retrieve relevant chunks for external LLMs."""
        if self.vector_store is None or not user_query:
            return ""
        try:
            docs = self.vector_store.similarity_search(user_query, k=top_k)
            return "\n---\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[RAG] Context retrieval error: {e}")
            return ""

    def query(self, user_query: str, top_k: int = 3) -> str:
        """
        Executes the RAG sequence:
        1. Embeds query & searches FAISS.
        2. Retrieves top_k chunks.
        3. Prompts local Ollama LLM.
        """
        # Error handling: Empty Index
        if self.vector_store is None:
            return "My knowledge base is completely empty. Please ingest documents first."
        
        if not user_query or not user_query.strip():
            return "I didn't hear a question."

        try:
            # 1. Similarity Search Fast Retrieval
            retrieved_docs = self.vector_store.similarity_search(user_query, k=top_k)
            
            # 2. Format context
            context_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            
            # 3. Construct Final Prompt
            formatted_prompt = self.prompt_template.format(
                context=context_text,
                question=user_query
            )

            if not self.llm:
                return "My Ollama inference engine is offline, but my database is intact."
                
            # 4. Inference via Ollama
            # Ensures failure handled gracefully if Ollama daemon is offline
            response = self.llm.invoke(formatted_prompt)
            return response.strip()

        except ConnectionError:
            return "I am currently unable to reach my language processing engine. Please ensure Ollama is running."
        except Exception as e:
            print(f"[RAG] Query Pipeline Error: {e}")
            return "An internal error occurred while processing my knowledge base."

# Quick test execution
if __name__ == "__main__":
    # Test Initialization
    rag = RAGPipeline()
    
    # Ingestion test (create dummy file if none exists)
    test_doc = Path("data/docs/test.txt")
    if not test_doc.exists():
        test_doc.parent.mkdir(parents=True, exist_ok=True)
        test_doc.write_text("The secret voice assistant codeword is 'alpha tango'.")
        
    rag.ingest_documents()
    
    # Query test
    print("\n[User]: What is the secret codeword?")
    ans = rag.query("What is the secret codeword?")
    print(f"[Assistant]: {ans}")
