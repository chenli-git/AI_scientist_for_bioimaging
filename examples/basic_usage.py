"""
Basic Usage Example for AI Bio Agent
=====================================

This example shows the simplest way to use aibioagent.
"""

import aibioagent as aba

def main():
    print("="*60)
    print("AI Bio Agent - Basic Usage Example")
    print("="*60)
    
    # Step 1: Setup (one-time)
    print("\n1. Setting up API key...")
    api_key = input("Enter your OpenAI API key: ")
    aba.set_api_key(api_key)
    
    # Step 2: Add papers (optional)
    print("\n2. Adding papers to knowledge base...")
    add_papers = input("Do you have PDFs to add? (y/n): ").lower()
    
    if add_papers == 'y':
        pdf_folder = input("Enter path to PDF folder: ")
        collection_name = input("Enter collection name (or press Enter for 'bioimage_segmentation'): ")
        collection_name = collection_name or "bioimage_segmentation"
        
        aba.add_papers(pdf_folder, collection=collection_name)
    
    # Step 3: List available collections
    print("\n3. Available knowledge bases:")
    collections = aba.list_collections()
    
    # Step 4: Ask questions
    print("\n4. Interactive Q&A")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("\n💬 Your question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question.strip():
            continue
        
        # Check if user wants to analyze an image
        if any(word in question.lower() for word in ['image', 'analyze', 'workflow']):
            image_prompt = input("Do you have an image to analyze? (path or press Enter to skip): ")
            image_path = image_prompt if image_prompt.strip() else None
        else:
            image_path = None
        
        # Get response
        print("\n🤖 AI Agent:")
        try:
            response = aba.ask(question, image_path=image_path)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print("Session complete!")
    print("="*60)


if __name__ == "__main__":
    main()
