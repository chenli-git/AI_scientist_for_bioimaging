# core/llm_client.py
from langchain_openai import ChatOpenAI
from config.settings import OPENAI_API_KEY, LLM_MODEL

def get_llm(temperature=0.2):
    """Return a ready-to-use LLM client."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=temperature
    )
    return llm

# Quick Test
def main():
    llm = get_llm()
    question = "who is julia roberts."
    response = llm.invoke(question)
    print("Question:", question)
    print("Response:", response.content)

if __name__ == "__main__":
    main()

