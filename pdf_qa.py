from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

from torch import chunk

load_dotenv(override=True)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions from a PDF document.
Use ONLY the context below to answer. If the answer is not in
the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:
""")

def load_pdf(pdf_path):
    """Load PDF and split into chunks."""
    print(f"\nLoading PDF: {pdf_path}")

    loader=PyPDFLoader(pdf_path)
    pages=loader.load()
    print(f"Loaded {len(pages)} pages")

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks= splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    return chunks

def create_vectorstore(chunks):
    """Embed chunks and store in FAISS."""
    print("Creating embeddings... (first time takes ~30 seconds)")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector store redy!")
    return vectorstore

def create_chain(vectorstore):
    """Build the RAG chain."""
    retriever=vectorstore.as_retriever(
        search_kwarge={"k":3}
    )

    def format_docs(docs):
       return "\n\n".join(doc.page_content for doc in docs)

    chain = (
          {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    |StrOutputParser()
    )
    return chain

def main():
    print("PDF Q&A BOT")
    print("-" * 40)

    pdf_path = input("Enter path to PDF: ").strip()

    pdf_path =pdf_path.strip('"').strip("'")

    if not os.path.isfile(pdf_path):
        print("File not found: {pdf_path}. Please check the path and try again.")
        return
    
    chunks = load_pdf(pdf_path)
    vectorstore = create_vectorstore(chunks)
    chain = create_chain(vectorstore)

    print("\nYou can now ask questions about the PDF. Type 'exit' to quit.")   
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            print("\nSearching documents...")
            answer = chain.invoke(question)
            print(f"\nBot: {answer}\n")

            print("-" * 40)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()