from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from core.embeddings import get_vectorstore
from core.llm_client import get_llm
from .base_agent import BaseAgent
from config.prompts.scientist_prompt import SCIENTIST_PROMPT

from core.debug_utils import debug_stage
from core.memory_manager import GLOBAL_MEMORY