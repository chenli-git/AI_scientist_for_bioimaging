"""
AI Bio Agent
============

A multi-agent AI system for biomedical imaging research.

Quick Start:
-----------
    import aibioagent as aba
    
    # Setup
    aba.quickstart(api_key="sk-your-key")
    
    # Ask questions
    response = aba.ask("What is adaptive optics?")
    print(response)

For detailed documentation, see USER_GUIDE.md or:
    >>> import aibioagent as aba
    >>> help(aba)
"""

# Import main user-facing API
from aibioagent import (
    # Configuration
    set_api_key,
    get_api_key,
    set_llm_model,
    set_vision_model,
    set_embed_model,
    get_models,
    info,
    quickstart,
    
    # Knowledge Base Management
    add_papers,
    add_urls,
    get_default_urls,
    list_collections,
    search_collection,
    delete_collection,
    
    # Query & Chat
    ask,
    chat,
    
    # Advanced - Direct Access
    get_scientist_agent,
    get_image_analyst,
    get_paper_reviewer,
    get_router,
    
    # Version
    __version__,
)

__all__ = [
    # Configuration
    "set_api_key",
    "get_api_key",
    "set_llm_model",
    "set_vision_model",
    "set_embed_model",
    "get_models",
    "info",
    "quickstart",
    
    # Knowledge Base
    "add_papers",
    "add_urls",
    "get_default_urls",
    "list_collections",
    "search_collection",
    "delete_collection",
    
    # Query
    "ask",
    "chat",
    
    # Advanced
    "get_scientist_agent",
    "get_image_analyst",
    "get_paper_reviewer",
    "get_router",
    
    # Metadata
    "__version__",
]
