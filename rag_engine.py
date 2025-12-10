import numpy as np
import requests
from openai import OpenAI
import pypdf
import chromadb
import uuid

class MinimalRAG:
    def __init__(self, ollama_base_url="http://64.226.86.247:11434", llm_base_url="https://procia-models2.winu.fr/v1", embedding_model="nomic-embed-text:v1.5", llm_model="Qwen/Qwen3-4B-Instruct-2507-FP8", api_key="k", collection_name="rag_collection", persist_directory="./chroma_db"):
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key
        )
        
        # Initialize ChromaDB (Persistent)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.history = [] # Stores last 5 Q&A pairs (10 messages)

    def clear_collection(self):
        """Clears the current collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    def extract_text(self, uploaded_file):
        """Extracts text from uploaded PDF or TXT file."""
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            return uploaded_file.getvalue().decode("utf-8")

    def chunk_text(self, text, chunk_size=800, overlap=80):
        """Chunks text with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def get_embedding(self, text):
        """Get embedding from Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 768 # Fallback

    def ingest(self, uploaded_file):
        """Full ingestion pipeline: Extract -> Chunk -> Embed -> Store in Chroma."""
        text = self.extract_text(uploaded_file)
        chunks = self.chunk_text(text)
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = [self.get_embedding(chunk) for chunk in chunks]
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        return len(chunks)

    def retrieve(self, query, top_k=3):
        """Retrieve top_k most similar chunks using ChromaDB."""
        query_embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # Chroma returns list of lists
        return results['documents'][0] if results['documents'] else []

    def update_history(self, role, content):
        """Update chat history, keeping only last 5 exchanges (10 messages)."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def generate(self, query, context_chunks):
        """Generate response using OpenAI client with history and context."""
        context = "\n\n".join(context_chunks)
        
        # Format history for the prompt
        history_context = ""
        for msg in self.history:
            history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

        system_prompt = f"""You are a helpful assistant. Use the following context and conversation history to answer the question.
        
Context:
{context}

History:
{history_context}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            stream=True
        )
        return response
