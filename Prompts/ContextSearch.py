def Doc_Context_Prompt(context, query):
    return f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """