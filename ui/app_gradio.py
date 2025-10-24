"""
app_gradio.py
-------------
Gradio ChatGPT-style interface for AI Scientist Agent.
"""

import gradio as gr
from core.rag_pipeline import RAGPipeline
from agents.AI_scientist_agent import AIScientistAgent


# ------------------------------------------------------------------
# Initialize default RAG pipeline and agent
# ------------------------------------------------------------------
rag = RAGPipeline(agent_cls=AIScientistAgent)


# ------------------------------------------------------------------
# Chat function with streaming
# ------------------------------------------------------------------
def chat_stream(message, history):
    """Stream tokens from the AI Scientist agent in ChatGPT style."""
    user_query = message.strip()
    if not user_query:
        yield "Please type a question."
        return

    try:
        partial_text = ""
        for chunk in rag.agent.stream(user_query):
            partial_text += chunk
            yield partial_text
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"


# ------------------------------------------------------------
# Build ChatGPT-like interface
# ------------------------------------------------------------
def build_chat_interface():
    return gr.ChatInterface(
        fn=chat_stream,
        title="🧠 AI Scientist Agent",
        description=(
            "Ask scientific or biomedical research questions. "
            "The AI Scientist agent uses retrieval-augmented reasoning "
            "and domain expertise to generate evidence-based answers."
        ),
        theme=gr.themes.Soft(primary_hue="indigo"),
        examples=[
            ["What are the latest deep-learning models for neuron segmentation?"],
            ["Explain the role of mitochondrial metabolism in astrocyte-neuron coupling."],
            ["How does adaptive optics improve light-sheet microscopy?"],
        ],
        chatbot=gr.Chatbot(height=500, show_copy_button=True),
    )

# ------------------------------------------------------------
if __name__ == "__main__":
    demo = build_chat_interface()
    demo.launch(debug=True)