from Prompts.ContextSearch import Doc_Context_Prompt
from src.vector_db import get_context_by_query, intialize_vectorstore
from src.local_llms import llama3_llm


if __name__ == "__main__":
    DB = intialize_vectorstore("sample.pdf")

    print(llama3_llm.invoke("Say hello in one sentence."))

    # ......  GENERATE ANSWER ..........

    # ask question from the document like "what is the main topic of the document?""
    query = "What is the main topic of the document? how may pages are there in the document?"
    context = get_context_by_query(DB, query)
    promt = Doc_Context_Prompt(context, query)
    response = llama3_llm.invoke(promt)
    print("\n\n", response)





