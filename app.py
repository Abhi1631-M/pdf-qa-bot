import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os

load_dotenv(override=True)

st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .upload-box { 
        border: 2px dashed #ccc; 
        border-radius: 12px; 
        padding: 2rem;
        text-align: center;
    }
    .source-box {
        background: #f8f9fa;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ── LLM and Embeddings ────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

# ── Process PDF ───────────────────────────────────────────
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading PDF..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

    with st.spinner(f"Splitting {len(pages)} pages into chunks..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

    with st.spinner(f"Creating embeddings for {len(chunks)} chunks..."):
        embeddings = load_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    os.unlink(tmp_path)
    return vectorstore, len(pages), len(chunks)

def create_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions from a PDF document.
Use ONLY the context below to answer. If the answer is not in
the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ── UI Layout ─────────────────────────────────────────────
st.title("📄 PDF Q&A Bot")
st.caption("Upload a PDF and ask questions about it")

# Sidebar — PDF upload
with st.sidebar:
    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["pdf"],
        help="Upload any PDF file to start asking questions"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            # New PDF uploaded — process it
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []  # clear chat history

            try:
                vectorstore, pages, chunks = process_pdf(uploaded_file)
                chain, retriever = create_chain(vectorstore)
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.success(f"Ready! {pages} pages, {chunks} chunks")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    if st.session_state.pdf_name:
        st.divider()
        st.markdown(f"**Current file:**")
        st.markdown(f"📄 {st.session_state.pdf_name}")

        if st.button("Upload different PDF"):
            st.session_state.pdf_name = None
            st.session_state.chain = None
            st.session_state.messages = []
            st.session_state.retriever = None
            st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Upload any PDF
    2. It gets split into chunks
    3. Chunks are embedded into vectors
    4. Your question finds relevant chunks
    5. Groq answers from those chunks
    """)

# ── Main chat area ────────────────────────────────────────
if not st.session_state.chain:
    # No PDF uploaded yet — show welcome screen
    st.markdown("### Get started")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Step 1**\nUpload a PDF from the sidebar")
    with col2:
        st.info("⚙️ **Step 2**\nWait for processing")
    with col3:
        st.info("💬 **Step 3**\nAsk any question")

else:
    # PDF loaded — show chat
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View sources from PDF"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Chunk {i+1} 
                        (Page {source.metadata.get('page', '?') + 1})</strong><br>
                        {source.page_content[:200]}...
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask a question about the PDF..."):

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    # Get answer
                    answer = st.session_state.chain.invoke(question)

                    # Get source chunks
                    sources = st.session_state.retriever.invoke(question)

                    st.markdown(answer)

                    # Show sources
                    with st.expander("View sources from PDF"):
                        for i, source in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-box">
                            <strong>Chunk {i+1} 
                            (Page {source.metadata.get('page', '?') + 1})</strong><br>
                            {source.page_content[:200]}...
                            </div>
                            """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {e}")