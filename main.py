# entry point, Gradio
"""
Main entry point for the AI Scientist Agent project.
Author: Chen Li
Date: 2025
"""

import argparse
from core.rag_pipeline import RAGPipeline
from agents.AI_scientist_agent import AIScientistAgent
# from agents.PaperReviewerAgent import PaperReviewerAgent  # example future agent

def run_cli():
    """Simple command-line chat loop."""
    rag = RAGPipeline(agent_cls=AIScientistAgent)
    print("🧠 AI Scientist CLI mode. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        try:
            response = ""
            for chunk in rag.agent.stream(query):
                print(chunk, end="", flush=True)
                response += chunk
            print("\n")  # newline after streaming output
        except Exception as e:
            print(f"⚠️ Error: {e}\n")


# Import UI only when needed (so HPC jobs don’t require Gradio)
def run_gradio():
    from ui.app_gradio import build_chat_interface
    demo = build_chat_interface()
    demo.launch(debug=True)

def main():
    parser = argparse.ArgumentParser(description="AI Scientist Multi-Agent System")
    parser.add_argument(
        "-m", "--mode",
        choices=["gradio", "cli"],
        default="gradio",
        help="Choose how to run the app: gradio (web UI) or cli (terminal)",
    )
    args = parser.parse_args()

    if args.mode == "gradio":
        run_gradio()
    elif args.mode == "cli":
        run_cli()


if __name__ == "__main__":
    main()