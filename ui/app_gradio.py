import gradio as gr
from core.rag_pipeline import build_rag_chain

# Build the RAG chain once
rag_chain = build_rag_chain()

def chat_with_rag(user_input):
    """Run a simple user query through the RAG pipeline."""
    try:
        result = rag_chain.invoke(user_input)
        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
# 🔹 Streaming response function
# ------------------------------
def stream_response(user_input):
    """Stream tokens from the Runnable RAG chain as they are generated."""
    try:
        # Yield tokens one by one
        for chunk in rag_chain.stream(user_input):
            if isinstance(chunk, str):
                yield chunk
            elif isinstance(chunk, dict) and "answer" in chunk:
                yield chunk["answer"]
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

# ------------------------------
# 🔹 Chatbot logic
# ------------------------------
def user_message(message, history):
    """Immediately show user's message, then wait for model output."""
    history = history + [[message, None]]  # Show message instantly
    return "", history  # Clear input box, keep history updated


def bot_response(history):
    """Stream model output to the latest user message."""
    user_query = history[-1][0]  # Get the most recent user question
    bot_message = ""  # Collect streamed tokens

    for chunk in stream_response(user_query):
        bot_message += chunk
        history[-1][1] = bot_message  # Update the bot's latest message
        yield history  # Stream update to UI



# ------------------------------
# 🔹 Launch Gradio interface
# ------------------------------
def main():
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown(
            "# 🧬 AI Scientist RAG Chatbot (Runnable Pipeline)\n"
            "Ask questions grounded in your biomedical papers. "
            "Built with **LangChain Runnables + ChromaDB + OpenAI API**."
        )

        chatbot = gr.Chatbot(height=450, label="AI Scientist")
        msg = gr.Textbox(
            placeholder="Ask me about biomedical image segmentation...",
            label="Your Question",
        )
        clear = gr.Button("🧹 Clear Chat")

        # When user hits Enter → instantly show message, then trigger streaming bot
        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(debug=True)

if __name__ == "__main__":
    main()

