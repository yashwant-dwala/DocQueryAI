from PyPDF2 import PdfReader

# # 3rd party libraries require key
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# local model (no key required)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
import os
load_dotenv()



# ......  LOAD & CHUNCKS DOC ..........
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

text = load_pdf("sample.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# ......  CREATE EMBEDDINGS & STORE IN FAISS DB ..........
print(os.getenv("OPENAI_API_KEY"))
# embeddings = OpenAIEmbeddings()

# Local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_texts(chunks, embeddings)


# ......  GENERATE ANSWER ..........
# llm = ChatOpenAI(temperature=0)
llm = Ollama(
    temperature = 0,
    model="llama3",
    base_url="http://localhost:8003"
)

print(llm.invoke("Say hello in one sentence."))

while True:
    # ask question from the document like "what is the main topic of the document?""
    query = input("Ask a question from the document or press 1 to exit: \n")

    if query == "1":
        print("Exiting...")
        break

    # ......  SIMILARITY SEARCH for QUESTION ..........
    docs = vectorstore.similarity_search(query, k=3) # k is the number of chunks to retrieve. (3-5)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    print(response, "\n\n")


