# entry point, Gradio
"""
Main entry point for the AI Scientist Agent project.
Author: Chen Li
Date: 2025
"""

import argparse
import uuid
from core.rag_pipeline import RAGPipeline
from agents.AI_scientist_agent import AIScientistAgent
from core.memory_manager import MemoryManager
# from agents.PaperReviewerAgent import PaperReviewerAgent  # example future agent

def run_cli(agent_cls, use_manager: bool = True):
    """Run the AI Scientist in interactive CLI mode."""
    rag = RAGPipeline(agent_cls=agent_cls)
    memory_manager = MemoryManager()
    session_id = "cli-session-" + uuid.uuid4().hex[:8]
     # Simple local transcript for contextualization
    conversation_log = []
    print(f"🧠 {agent_cls.__name__} (session: {session_id})")
    print("Type 'exit' to quit.\n")

    while True:
        query = input(">>> ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            break
        try:
            history_text = "\n".join(
                [f"User: {u}\nAI: {a}" for u, a in conversation_log]
            )
            if use_manager and memory_manager:
                contextualized_query = memory_manager.contextualize(query, history_text)
            else:
                contextualized_query = query
            response = rag.agent.run(contextualized_query, session_id=session_id)
            print(f"\n🧩 {response}\n")
            conversation_log.append((query, response))
        except Exception as e:
            print(f"⚠️ Error: {e}\n")



# Import UI only when needed (so HPC jobs don’t require Gradio)
def run_gradio(agent_cls):
    from ui.app_gradio import build_interface
    demo = build_interface(agent_cls=agent_cls)
    demo.launch(debug=True)

def main():
    parser = argparse.ArgumentParser(description="AI Scientist Multi-Agent System")
    parser.add_argument(
        "-m", "--mode",
        choices=["gradio", "cli"],
        default="gradio",
        help="Choose how to run the app: gradio (web UI) or cli (terminal)",
    )

    parser.add_argument(
        "-a", "--agent",
        type=str,
        choices=["scientist"],  # Add 'reviewer', 'analyst', etc. later
        default="scientist",
        help="Select which agent to use"
    )
    args = parser.parse_args()

    # Map CLI flag to class
    agent_map = {
        "scientist": AIScientistAgent,
        # "reviewer": PaperReviewerAgent,
    }
    agent_cls = agent_map.get(args.agent)


    if args.mode == "gradio":
        run_gradio(agent_cls)
    elif args.mode == "cli":
        run_cli(agent_cls)

if __name__ == "__main__":
    main()