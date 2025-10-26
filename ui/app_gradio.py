"""
app_gradio.py
-------------
Gradio ChatGPT-style interface for AI Scientist Agent.
"""

import gradio as gr
import uuid
from core.rag_pipeline import RAGPipeline
from agents.AI_scientist_agent import AIScientistAgent
from core.memory_manager import MemoryManager


# ------------------------------------------------------------
# Build ChatGPT-like interface
# ------------------------------------------------------------
def build_interface(agent_cls):
    """Construct the streaming Gradio Chat UI for the selected agent."""
    rag = RAGPipeline(agent_cls=AIScientistAgent)
    memory_manager = MemoryManager()

    def stream_agent(user_query, history):
        """Handles streaming responses while keeping contextualized memory."""
        if not user_query.strip():
            yield "Please enter a question."
            return

        # Convert history list of [user, ai] pairs → readable text
        history_text = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history]) if history else ""

        # Contextualize the query using memory
        contextualized_query = memory_manager.contextualize(user_query, history_text)

        # Each chat window gets its own session_id
        session_id = "gradio-session-" + uuid.uuid4().hex[:8]

        output = ""
        for chunk in rag.agent.stream(contextualized_query, session_id=session_id):
            output += chunk
            yield output


    return gr.ChatInterface(
        fn=stream_agent,
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
    demo = build_interface()
    demo.launch(debug=True)