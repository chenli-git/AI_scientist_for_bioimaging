from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from core.embeddings import get_vectorstore
from core.llm_client import get_llm
from .base_agent import BaseAgent
from config.prompts.scientist_prompt import SCIENTIST_PROMPT

class AIScientistAgent(BaseAgent):
    """
    AI Scientist Agent
    ------------------
    This agent performs retrieval-augmented generation (RAG)
    for scientific reasoning and literature question answering.

    Responsibilities:
    - Defines its own prompt
    - Initializes RAG pipeline (retriever + LLM)
    - Exposes a simple `.run(query)` interface
    """
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        # 1️⃣ Load vectorstore for retrieval
        self.vectorstore = get_vectorstore("bioimage_segmentation")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        # 2️⃣ Load LLM client
        self.llm = get_llm(temperature=temperature)
        # 3️⃣ Load predefined prompt from config/prompts
        self.prompt = ChatPromptTemplate.from_template(SCIENTIST_PROMPT)
        # 4️⃣ Build retrieval-augmented pipeline
        self.rag_chain = (
            {"context": self.retriever | self._combine_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _combine_docs(self, docs):
        """Merge retrieved documents into a single context string."""
        return "\n\n".join([d.page_content for d in docs])
    
    def run(self, query: str) -> str:
        """Run RAG pipeline for a given query."""
        return self.rag_chain.invoke(query)
    
    def stream(self, query: str):
        """Stream RAG responses token by token."""
        for chunk in self.rag_chain.stream(query):
            yield chunk
        

















    






