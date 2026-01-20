from langchain_ollama import OllamaLLM

llama3_llm = OllamaLLM(
    temperature = 0,
    model="llama3",
    base_url="http://localhost:8003"
)

phi3_llm = OllamaLLM(
    temperature = 0,
    model="phi3",
    base_url="http://localhost:8003"
)

deepseek_llm = OllamaLLM(
    temperature = 0,
    model="deepseek-llm:7b",
    base_url="http://localhost:8003"
)