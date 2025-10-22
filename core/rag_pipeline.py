from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from core.embeddings import get_vectorstore
from core.llm_client import get_llm

def build_rag_chain():
    """Build a simple RAG pipeline using LangChain Runnables."""
    llm = get_llm()
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})
    # Prompt: specialized for biomedical context
    prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. use the following pieces of retrieved context to answer the question if the context has relevant infomation. 
        if you dont know the answer, just say that you dont't know.

        Context:
        {context}

        Question:
        {question}
        """)
    
    # Runnable chain: retriever → prompt → LLM → output
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain



if __name__ == "__main__":
    a = 1