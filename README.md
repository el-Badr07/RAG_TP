# RAG Assistant

A minimal, production-ready Retrieval-Augmented Generation (RAG) system that combines document retrieval with LLM-powered responses. Upload documents, ask questions, and get contextually accurate answers with cited sources.

## Features

- **Document Ingestion** – Upload PDF and TXT files with automatic chunking and embedding
- **Vector Storage** – Persistent ChromaDB with semantic search capability
- **Multi-Model Support** – Flexible embedding and LLM model configuration
- **Conversation Memory** – Maintains last 5 Q&A exchanges for contextual responses
- **Streaming Responses** – Real-time response generation with visual feedback
- **Clean UI** – Minimalist Streamlit interface with dark theme

## Architecture

```
User Input
    ↓
[Retrieval] → ChromaDB (Semantic Search)
    ↓
[Context + History] → LLM (OpenAI-compatible API)
    ↓
[Streaming Response]
```

### Core Components

**MinimalRAG** (`rag_engine.py`)
- Document extraction (PDF/TXT)
- Text chunking with overlap
- Embedding generation via Ollama
- Vector database management
- Query retrieval and LLM streaming

**Streamlit App** (`app.py`)
- Interactive chat interface
- Settings panel for model configuration
- Document upload and ingestion
- Real-time streaming display

## Installation

### Prerequisites
- Python 3.8+
- Ollama running  (for embeddings)
- Access to an OpenAI-compatible LLM API

### Setup

1. **Clone and enter directory**
   ```bash
   cd RAG-sl
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Environment Variables / Settings
Configure these in the Streamlit sidebar:

| Setting | Default | Purpose |
|---------|---------|---------|
| **Embedding Model** | `nomic-embed-text:v1.5` | Ollama embedding model |
| **LLM Model** | `Qwen/Qwen3-4B-Instruct-2507-FP8` | LLM for response generation |
| **Ollama URL** | `http://ip_address:11434` | Ollama API endpoint |
| **LLM Base URL** | `https://base_url/v1` | OpenAI-compatible API endpoint |
| **API Key** | `k` | Authentication token |
| **Collection Name** | `rag_collection` | ChromaDB collection identifier |
| **Retrieved Chunks** | `3` | Number of context chunks for retrieval |

## Usage

### Running the Application

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501`

### Workflow

1. **Upload Document** – Use sidebar file uploader (PDF or TXT)
2. **Ingest Data** – Click "Ingest Data" button to extract, chunk, and embed
3. **Ask Questions** – Type queries in the chat input
4. **Get Answers** – System retrieves relevant chunks and generates responses

## Technical Details

### Data Processing Pipeline

**Ingestion**
1. Extract text from uploaded document
2. Split into chunks (800 chars, 80-char overlap)
3. Generate embeddings for each chunk via Ollama
4. Store in ChromaDB with cosine similarity metric

**Retrieval**
1. Embed user query
2. Semantic search in ChromaDB (top-k most similar)
3. Pass chunks + conversation history to LLM

**Generation**
1. Construct system prompt with context and history
2. Stream response from LLM API
3. Display incrementally in UI

### Key Configuration Options

**Chunking Strategy**
- `chunk_size=800` – Characters per chunk
- `overlap=80` – Overlap between chunks to preserve context

**Retrieval**
- Cosine similarity metric (HNSW index)
- Configurable top-k results (1-10 chunks)

**Memory**
- Maintains last 5 Q&A pairs (10 messages)
- Automatically truncates older exchanges

## Database

**ChromaDB Storage**
- Location: `./chroma_db/`
- Type: Persistent (SQLite)
- Collections: Named by user configuration
- Clear collection anytime via sidebar
