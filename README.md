# DocQueryAI
Retrieval-Augumented Generation of Answers from the provided doc using 

## Installation
pip install openai faiss-cpu streamlit PyPDF2 langchain



UI used is OpenWebUI
LLM used are locally downloaded models in ollama (deepseek-llm:7b for code and llama3:latest for Q&A)             
Embeddings are created using OpenAI (can use SentenceTransformers locally)
Vector DB used is FAISS from langchain (later can use Chroma DB)


