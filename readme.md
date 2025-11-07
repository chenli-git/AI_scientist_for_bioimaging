# AI Scientist System for Biomedical Imaging
The **AI Scientist** project is a modular, multi-agent framework that unifies **retrieval-augmented reasoning**, **conversational memory**, and **image understanding** to accelerate research in **biomedical imaging**.  
It enables scientists to ask natural-language questions, analyze microscopy data, and design reproducible image-processing workflows — all within a single intelligent system.

---

## 🧩 Overview

This system is built around specialized AI "agents," each designed for a specific research task:
- **AI_scientist_agent.py** → text-based RAG for scientific Q&A
- **Image_analyst_agent.py** → multimodal vision + RAG for workflow design
- **paper_reviewer_agent.py** → PDF analysis + RAG for paper review
- **Router** → intelligent routing based on query intent + shared memory
- **GLOBAL_MEMORY** → unified conversation context across agents

| Agent | Primary Function |
|--------|------------------|
| **AI Scientist Agent** | Literature-grounded scientific reasoning via RAG |
| **ImageAnalyst Agent** | Workflow generation and interpretation of microscopy images |
| **PaperReviewer Agent** | Scientific paper analysis, critique, and literature review with PDF support |

Each agent is implemented as a composable LangChain `Runnable` pipeline with shared memory, individual prompt templates, and retrieval logic.  
The architecture is fully extensible — future agents (e.g., `DataAnalystAgent`, or `ModelTrainerAgent`) can be added easily.

---

## AI Scientist Agent

### **Description**
The **AI Scientist Agent** is the core reasoning engine of the system.  
It integrates a **Large Language Model (LLM)** with a **Retrieval-Augmented Generation (RAG)** pipeline that queries biomedical literature and microscopy papers stored in a local vector database.  
The main puprose is to provide Q&A of the local scientific papers and template for future agents.

### **Key Features**
- **Retrieval-Augmented Reasoning** – Combines LLM inference with document retrieval from a Chroma vector store.  
- **Session-Aware Memory** – Maintains multi-turn conversation context via `RunnableWithMessageHistory`.  
- **Streaming Output** – Supports live token streaming in Gradio for interactive dialogue.  
- **Contextual Query Rewriting** – Uses a `MemoryManager` to make follow-up questions self-contained.  
- **Paper Insights** – Answers are sourced from self-collected papers and technical documentation.

## ImageAnalyst Agent
### **Description**
The **ImageAnalyst Agent** bridges raw microscopy data and AI-assisted workflow design.
It reads uploaded images, extracts metadata and intensity statistics, and proposes step-by-step Fiji or Python analysis pipelines tailored to the data’s characteristics.


### **Key Features**
- **Raw Image Understanding** - Accepts microscopy images.
- **Workflow Recommendation** - Suggests details Fiji or python pipeliness.
- **RAG-based Fiji Knowledge** - Retrieves plugin documentation and tutorials from a continuously updated Fiji and other open source packages knowledge base.
1. Could accept two inputs, raw image, the user goal/question/description, optionally include summary
2. vision-capable LLM
3. searches both databases (tech docs and scientific papers)
4. return: detailed fiji/python workflow, a rationale grounded in both the image and context.


## PaperReviewer Agent
### **Description**
The **PaperReviewer Agent** specializes in scientific paper analysis, critique, and literature review for bioimaging research.  
It can read and analyze uploaded PDF papers, extracting text, tables, and figures, then provide comprehensive reviews grounded in both the paper content and relevant literature from the knowledge base.

### **Key Features**
- **PDF Content Extraction** – Automatically parses uploaded papers to extract full text, tables, and figure captions using Docling.
- **Literature-Grounded Reviews** – Combines uploaded paper content with RAG retrieval from the scientific literature database.
- **Critical Analysis** – Evaluates methodology, experimental design, novelty, and scientific rigor.
- **Constructive Feedback** – Provides evidence-based suggestions for improvement.
- **Session Memory** – Remembers uploaded papers for follow-up questions in the same session.
- **Flexible Queries** – Works with or without PDF uploads for literature reviews and paper summaries.


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
- OpenAI text and image 
- Langchain
- ChromaDB
- Gradio