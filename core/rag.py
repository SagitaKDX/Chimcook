import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import warnings
# Suppress noisy warnings from huggingface
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
# Prevent PyTorch from hogging cores needed by the LLM and STT
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from functools import lru_cache

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except ImportError:
    raise ImportError(
        "Missing RAG dependencies. Please install: "
        "pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu"
    )

import numpy as np
from langchain_core.embeddings import Embeddings

class NativeONNXEmbeddings(Embeddings):
    """A bulletproof, dependency-free LangChain wrapper for pure ONNX Runtime embeddings."""
    def __init__(self, model_path: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        # Xenova model specifies model_quantized.onnx or model.onnx
        onnx_file = f"{model_path}/model_quantized.onnx"
        if not os.path.exists(onnx_file):
            onnx_file = f"{model_path}/model.onnx"
            
        self.session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def _encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
        
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64)
        }
        
        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]
        
        # Mean Pooling over attention mask
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.repeat(attention_mask[:, :, np.newaxis], token_embeddings.shape[2], axis=2)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        
        # L2 Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.tolist()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)
        
    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]

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
        embedding_model_name: str = "models/embed_onnx", # Native ONNX explicitly built Path
        ollama_model: str = "llama3.2:1b", # Often exact tag in Ollama
        chunk_size: int = 600,
        chunk_overlap: int = 150
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
        self.embeddings = NativeONNXEmbeddings(
            model_path=embedding_model_name
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

    @lru_cache(maxsize=128)
    def _cached_similarity_search(self, user_query: str, top_k: int) -> str:
        if self.vector_store is None:
            return ""
        try:
            results = self.vector_store.similarity_search_with_score(user_query, k=top_k)
            # Filter matches with L2 distance score > 0.35 (too far)
            valid_docs = [doc.page_content for doc, score in results if score <= 0.35]
            if not valid_docs:
                return ""
            return "\n---\n".join(valid_docs)
        except Exception as e:
            print(f"[RAG] Context retrieval error: {e}")
            return ""

    def get_context(self, user_query: str, top_k: int = 2) -> str:
        """Lightweight method to just retrieve relevant chunks for external LLMs."""
        if self.vector_store is None or not user_query:
            return ""
        return self._cached_similarity_search(user_query.strip().lower(), top_k)

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
            results = self.vector_store.similarity_search_with_score(user_query, k=top_k)
            
            # 2. Format context
            valid_docs = [doc.page_content for doc, score in results if score <= 0.35]
            context_text = "\n---\n".join(valid_docs) if valid_docs else "No relevant context found in knowledge base."
            
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
