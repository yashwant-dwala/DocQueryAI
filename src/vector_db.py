from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import os

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def make_chuncks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return chunks


def get_context_by_query(vectorstore, query, k=4):
    docs = vectorstore.similarity_search(query, k=k) # k is the number of chunks to retrieve. (3-5)
    print("-INFO: Done Similarity Search for query in vectorstore")
    context = "\n".join([doc.page_content for doc in docs])
    print("-INFO: Context Prepared")
    return context


def intialize_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ......  LOAD & CHUNCKS DOC ..........
    if os.path.exists("src/faiss_index"):
        vectorstore = FAISS.load_local("src/faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("-INFO: Vectorstore loaded")
        return vectorstore
    
    text = load_pdf(path)
    print("-INFO: Text loaded from PDF")
    chunks = make_chuncks(text)
    print("-INFO: Text chunked")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("-INFO: Vectorstore created")
    vectorstore.save_local("src/faiss_index")
    print("-INFO: Vectorstore saved")
    return vectorstore


