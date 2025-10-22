# AI Scientist Agent — Biomedical Computer Vision
A Retrieval-Augmented Generation (RAG)–based AI assistant built using the OpenAI API and LangChain Runnables, designed to read, understand, and answer questions about biomedical imaging and segmentation research papers.

This project integrates ChromaDB for vector search and retrieval, LangChain 0.3+ Runnables for modular pipeline design, and an interactive Gradio UI for exploration.

# System Architecture
```text
[User Query]
     ↓
[Retriever: ChromaDB]
     ↓
[Prompt Template: Biomedical AI Scientist]
     ↓
[LLM: OpenAI GPT via API]
     ↓
[Output Parser: StrOutputParser]
     ↓
[Gradio UI Output]
```

# Quick Start
1. install dependencies
```
pip install -r requirements.txt
```
2. Add your OpenAI API key

#make a .env file and store your api key

3. Build your Chroma database
```
python -m data/document_loader # set the pdfs folder
```

# Techs
- OpenAI
- Langchain
- ChromaDB
- Gradio