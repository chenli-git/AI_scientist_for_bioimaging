"""
main.py
--------
Entry point for the AI Scientist application.

Supports two modes:
1. CLI mode  → command-line interactive loop
2. Gradio mode → web-based UI (default)

Usage:
    python main.py --mode cli
    python main.py --mode gradio

Requirements:
    pip install gradio Pillow
"""

import argparse
import sys
from core.router import Router

# Lazy import of Gradio UI to avoid loading heavy modules when in CLI
def run_gradio():
    try:
        # Adjust import path if app_gradio.py is in another directory
        from ui.app_gradio import build_interface
    except Exception as e:
        print(f"❌ Failed to import Gradio app: {e}")
        sys.exit(1)

    demo = build_interface()
    demo.queue(status_update_rate=0.1).launch(debug=True)


def run_cli():
    """
    Simple command-line chat loop for testing routing logic.
    """
    router = Router()
    print("🧠 AI Scientist CLI Mode")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("💬 Enter your query: ").strip()
            if query.lower() in ["exit", "quit"]:
                break

            img_path = input("🖼️ Optional image path (press Enter to skip): ").strip() or None
            pdf_path = input("📄 Optional PDF path (press Enter to skip): ").strip() or None

            response, label = router.route_query(
                query=query,
                session_id="cli_session",
                image_path=img_path,
                pdf_path=pdf_path,
            )
            print(f"\n➡ Routed to [{label.upper()}]\n")
            print(f"🧩 Response:\n{response}\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting gracefully...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="AI Scientist – Multi-Agent Assistant")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["cli", "gradio"],
        default="gradio",
        help="Choose how to run the app: cli or gradio",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    else:
        run_gradio()


if __name__ == "__main__":
    main()
    #python main.py -m cli

    # pip install --upgrade build twine # install build tolls
    # rm -rf build dist *.egg-info #remove old build
    # python -m build
    #  
