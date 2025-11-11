"""
Custom Database Example
=======================

Shows how to build custom knowledge bases for specific research areas.
"""

import aibioagent as aba

def main():
    # Setup
    aba.set_api_key("sk-your-key-here")
    
    print("Building custom research databases...\n")
    
    # Example 1: Cell Biology Database
    print("1. Adding cell biology papers...")
    aba.add_papers(
        "papers/cell_biology",
        collection="cell_bio",
        chunk_size=1000
    )
    
    # Example 2: Microscopy Techniques Database
    print("\n2. Adding microscopy papers...")
    aba.add_papers(
        "papers/microscopy",
        collection="microscopy_methods",
        chunk_size=1200
    )
    
    # Example 3: Web Documentation Database
    print("\n3. Adding online documentation...")
    urls = [
        "https://scikit-image.org/docs/stable/",
        "https://napari.org/stable/",
        "https://cellprofiler.org/",
    ]
    aba.add_urls(urls, collection="imaging_tools")
    
    # Example 4: Show all collections
    print("\n4. Available collections:")
    aba.list_collections()
    
    # Example 5: Query specific collections
    print("\n5. Searching specific collections...")
    
    query = "cell segmentation methods"
    
    for collection in ["cell_bio", "microscopy_methods", "imaging_tools"]:
        print(f"\nSearching in '{collection}':")
        results = aba.search_collection(query, collection, top_k=2)
        print(f"Found {len(results)} results")
    
    print("\n✅ Custom databases ready to use!")
    print("Now you can ask questions that leverage your specific knowledge bases.")


if __name__ == "__main__":
    main()
