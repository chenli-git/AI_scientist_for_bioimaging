# entry point, Gradio
"""
Main entry point for the AI Scientist Agent project.
Author: Chen Li
Date: 2025
"""

import argparse
from core.rag_pipeline import build_rag_chain
from ui.app_gradio import main as gradio_app

def run_cli():
    """
    Run in command-line mode for quick testing.
    Uses the RAG Runnable pipeline directly.
    """
    rag_chain = build_rag_chain()

    print("🧬 AI Scientist Agent (CLI Mode)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            result = rag_chain.invoke(user_input)
            print(f"AI: {result}\n")
        except Exception as e:
            print(f"⚠️ Error: {str(e)}\n")

if __name__ == "__main__":
    run_cli()