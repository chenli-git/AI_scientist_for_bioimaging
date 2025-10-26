"""
core/memory_manager.py
----------------------
Contextual query rewriting based on conversation history.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from core.llm_client import get_llm

class MemoryManager:
    """Uses the LLM to rewrite queries using previous conversation."""

    def __init__(self, temperature: float = 0.1):
        self.llm = get_llm(temperature=temperature)
        self.contextualizer_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that rewrites the latest user query
        so it makes sense without prior context.

        Conversation so far:
        {history}

        User's latest query:
        {query}

        Rewrite the user's query to be self-contained but preserve its intent.
        """)

        self.chain = (
            {"history": RunnablePassthrough(), "query": RunnablePassthrough()}
            | self.contextualizer_prompt
            | self.llm
            | StrOutputParser()
        )

    def contextualize(self, query: str, history_text: str) -> str:
        """Return a rewritten query that includes context from history."""
        if not history_text.strip():
            return query  # no history → no rewrite needed
        return self.chain.invoke({"history": history_text, "query": query})
    
if __name__ == "__main__":
    manager = MemoryManager()
    rewritten = manager.contextualize("child?", "User: how many cups of water a adult should drink?\nAI: ...")
    print(rewritten)

