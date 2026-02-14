from Prompts.ContextSearch import Doc_Context_Prompt
from src.vector_db import get_context_by_query, intialize_vectorstore
from src.local_llms import llama3_llm
from Helper.helper import Logger




if __name__ == "__main__":
    log = Logger()
    log.debug("........................Starting DocQueryAI Service..... ..... ... .. .")
    DB = intialize_vectorstore("sample.pdf")
    if DB is None:
        exit(1)

    log.debug(llama3_llm.invoke("Say hello in one sentence."))

    # ......  GENERATE ANSWER ..........

    # ask question from the document like "what is the main topic of the document?""
    query = "what is the main topic of the document?"
    context = get_context_by_query(DB, query)
    promt = Doc_Context_Prompt(context, query)
    response = llama3_llm.invoke(promt)
    log.info(response)





