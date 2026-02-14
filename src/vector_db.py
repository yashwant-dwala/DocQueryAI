from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from Helper.helper import Logger, load_pdf
log = Logger()


def make_chuncks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    return chunks


def get_context_by_query(vectorstore, query, k=4):
    log = Logger()
    docs = vectorstore.similarity_search(query, k=k) # k is the number of chunks to retrieve. (3-5)
    log.debug("Done Similarity Search for query in vectorstore")
    context = "\n".join([doc.page_content for doc in docs])
    log.debug("Context Prepared")
    return context


def intialize_vectorstore(path):
    log = Logger()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ......  LOAD & CHUNCKS DOC ..........
    if os.path.exists("src/faiss_index"):
        vectorstore = FAISS.load_local("src/faiss_index", embeddings, allow_dangerous_deserialization=True)
        log.debug("Vectorstore loaded")
        return vectorstore
    
    try:
        document = load_pdf(path)
        log.debug("document loaded from PDF")
        log.debug("Making Chunks....")
        chunks = make_chuncks(document)
        log.info(f"Created {len(chunks)} chunks")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        log.debug("Vectorstore created")
        vectorstore.save_local("src/faiss_index")
        log.debug("Vectorstore saved")
    except Exception as e:
        log.error(f"Error creating vectorstore: {e}")
        return None
    return vectorstore


