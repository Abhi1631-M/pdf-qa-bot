# PDF Q&A Bot 📄

Chat with any PDF using AI. Upload a document and ask questions — 
the bot answers only from your document with source references.

## Features
- Drag and drop PDF upload
- AI answers grounded in your document only
- Shows source chunks and page numbers for every answer
- Chat history within session
- Powered by Groq (Llama 3.3 70B) + FAISS vector search

## How RAG works
PDF → Split into chunks → Embed into vectors → 
Question finds relevant chunks → Groq answers from chunks

## Setup
1. Clone the repo
2. Create venv and activate
3. Install: `python -m pip install -r requirements.txt`
4. Add `GROQ_API_KEY` to `.env`
5. Run: `streamlit run app.py`

## Tech Stack
- Python 3.12
- LangChain
- Groq API — Llama 3.3 70B
- FAISS — vector similarity search
- HuggingFace Embeddings — sentence-transformers
- Streamlit — UI
