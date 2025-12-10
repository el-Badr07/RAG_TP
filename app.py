import streamlit as st
from rag_engine import MinimalRAG
import time

# Page Config
st.set_page_config(page_title="Minimal RAG", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Minimalist Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.2);
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #0d1117;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 6px;
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #1f2428;
        border-left: 4px solid #58a6ff;
    }
    
    .bot-message {
        background-color: #161b22;
        border-left: 4px solid #238636;
    }
    
    h1, h2, h3 {
        font-weight: 600;
        color: #f0f6fc;
    }
    
    .stSpinner > div {
        border-top-color: #238636 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "rag" not in st.session_state:
    st.session_state.rag = MinimalRAG()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    st.subheader("Models")
    embedding_model = st.text_input("Embedding Model", value="nomic-embed-text:v1.5")
    llm_model = st.text_input("LLM Model", value="Qwen/Qwen3-4B-Instruct-2507-FP8")
    
    st.subheader("Database")
    collection_name = st.text_input("Collection Name", value="rag_collection")
    top_k = st.slider("Retrieved Chunks", min_value=1, max_value=10, value=3)
    
    if st.button("Clear Collection", type="primary"):
        if st.session_state.rag.clear_collection():
            st.success(f"Collection '{collection_name}' cleared!")
        else:
            st.error("Failed to clear collection.")

    st.subheader("Connection")
    ollama_url = st.text_input("Ollama URL", value="http://ip_address:11434")
    llm_url = st.text_input("LLM Base URL", value="https://base_url/v1")
    api_key = st.text_input("API Key", value="k", type="password")
    
    # Update RAG instance if settings change
    if st.button("Update Settings"):
        st.session_state.rag = MinimalRAG(
            ollama_base_url=ollama_url,
            llm_base_url=llm_url,
            embedding_model=embedding_model,
            llm_model=llm_model,
            api_key=api_key,
            collection_name=collection_name
        )
        st.success("Settings updated!")

    st.divider()
    
    st.subheader("Data Ingestion")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Ingest Data"):
        with st.spinner("Ingesting and Chunking..."):
            try:
                num_chunks = st.session_state.rag.ingest(uploaded_file)
                st.success(f"Successfully ingested {num_chunks} chunks!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main Chat Interface
st.title("RAG Assistant")
# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask something about your document..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.rag.update_history("user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 1. Retrieve
        with st.status("Thinking...", expanded=False) as status:
            st.write("Retrieving relevant context...")
            context_chunks = st.session_state.rag.retrieve(prompt, top_k=top_k)
            st.write(f"Found {len(context_chunks)} relevant chunks.")
            status.update(label="Context Retrieved", state="complete", expanded=False)
            
        # 2. Generate
        try:
            stream = st.session_state.rag.generate(prompt, context_chunks)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.rag.update_history("assistant", full_response)
        except Exception as e:
            st.error(f"Generation Error: {str(e)}")
            full_response = "I encountered an error generating the response."
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
