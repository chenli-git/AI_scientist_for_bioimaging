# 📘 AI Bio Agent - User Guide

## Quick Start

### Installation
```bash
pip install aibioagent
```

### Basic Setup
```python
import aibioagent as aba

# Setup (one-time)
aba.quickstart(
    api_key="sk-your-openai-key",
    pdf_folder="path/to/your/papers"  # optional
)

# Ask questions
response = aba.ask("What is adaptive optics in microscopy?")
print(response)
```

---

## 📖 Complete API Reference

### 1. Configuration

#### Set API Key
```python
import aibioagent as aba

# Set for current session and save to .env
aba.set_api_key("sk-your-key-here")

# Set for session only (don't save)
aba.set_api_key("sk-your-key-here", save_to_env=False)

# Check current key
key = aba.get_api_key()
print(f"API key: {key[:10]}...")
```

---

### 2. Building Knowledge Bases

AI Bio Agent uses **two separate types of vector databases**:

1. **📄 Papers Database** - Research papers and publications (PDFs)
2. **💻 Code Documentation Database** - Online documentation and tutorials (URLs)

#### Understanding the Two Databases

**Papers Database (PDFs)**
- Purpose: Store scientific literature for RAG retrieval
- Format: PDF files (research papers, reviews, publications)
- Function: `add_papers()`
- Default collection: `"papers"`
- Example collections: `"microscopy_papers"`, `"crispr_papers"`, `"segmentation_papers"`

**Code Documentation Database (URLs)**
- Purpose: Store technical documentation and tutorials
- Format: Web pages (documentation sites, API references, tutorials)
- Function: `add_urls()`
- Default collection: `"code_docs"`
- Example collections: `"opencv_docs"`, `"napari_docs"`, `"scikit_image_docs"`
- **Built-in defaults**: ImageJ/Fiji, scikit-image, OpenCV, Pillow, LangChain docs

#### Add PDF Papers
```python
import aibioagent as aba

# Add a single PDF file
aba.add_papers("path/to/paper.pdf", collection="important_paper")

# Add all PDFs from a folder
aba.add_papers("papers/microscopy", collection="microscopy_papers")

# Multiple folders for different topics
aba.add_papers("papers/segmentation", collection="segmentation_papers")
aba.add_papers("papers/crispr", collection="crispr_papers")

# Custom chunk settings
aba.add_papers(
    "papers/general",
    collection="general_papers",
    chunk_size=500,      # Smaller chunks
    chunk_overlap=100,   # Less overlap
    verbose=True
)
```

#### Add Web Documentation
```python
import aibioagent as aba

# See what URLs are included by default
aba.get_default_urls()
# Shows: ImageJ, scikit-image, OpenCV, Pillow, LangChain docs

# Add single URL
aba.add_urls("https://napari.org/stable/", collection="napari_docs")

# Add multiple URLs
urls = [
    "https://scikit-image.org/docs/stable/",
    "https://docs.opencv.org/4.x/",
    "https://pillow.readthedocs.io/"
]
aba.add_urls(urls, collection="image_libs_docs")

# Add tool-specific documentation
aba.add_urls("https://cellprofiler.org/", collection="cellprofiler_docs")

# With crawling options
aba.add_urls(
    "https://docs.example.com",
    collection="my_tool_docs",
    max_depth=2,       # Crawl 2 levels deep
    max_pages=200,     # Max 200 pages per root
    verbose=True
)
```

---

### 3. Querying

#### Basic Questions
```python
import aibioagent as aba

# Literature questions (uses RAG)
response = aba.ask("What are the latest CRISPR techniques?")

# Technical questions
response = aba.ask("How do I segment nuclei in fluorescence microscopy?")

# Methodology questions
response = aba.ask("Compare watershed vs Otsu segmentation")
```

#### Image Analysis
```python
import aibioagent as aba

# Analyze an image
response = aba.ask(
    question="What segmentation method should I use for these cells?",
    image_path="microscopy_image.tif"
)

# Get workflow suggestions
response = aba.ask(
    question="Suggest a complete image analysis workflow",
    image_path="sample.png"
)
```

#### Paper Review
```python
import aibioagent as aba

# Review a paper
response = aba.ask(
    question="Summarize the methodology and findings",
    pdf_path="research_paper.pdf"
)

# Critique a paper
response = aba.ask(
    question="What are the strengths and weaknesses?",
    pdf_path="paper.pdf"
)
```

#### Streaming Responses
```python
import aibioagent as aba

# Stream response token by token
for chunk in aba.ask("Explain deep learning for microscopy", stream=True):
    print(chunk, end="", flush=True)
```

---

### 4. Interactive Chat

#### Terminal Chat
```python
import aibioagent as aba

# Start CLI chat
aba.chat()  # or aba.chat(mode="cli")
```

#### Web UI Chat
```python
import aibioagent as aba

# Start Gradio web interface
aba.chat(mode="gradio")
```

---

### 5. Database Management

#### List Collections
```python
import aibioagent as aba

# Show all collections
collections = aba.list_collections()
# Output:
# 📚 Available collections (3):
#   • bioimage_segmentation
#   • online_tech_docs
#   • my_papers
```

#### Search Collections
```python
import aibioagent as aba

# Search in a specific collection
results = aba.search_collection(
    query="cell segmentation methods",
    collection="bioimage_segmentation",
    top_k=5
)

# Access results
for result in results:
    print(f"Source: {result['source']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
```

#### Delete Collections
```python
import aibioagent as aba

# Safety check - shows warning
aba.delete_collection("old_collection")

# Actually delete
aba.delete_collection("old_collection", confirm=True)
```

---

### 6. Advanced Usage

#### Direct Agent Access
```python
import aibioagent as aba

# Get specific agents
scientist = aba.get_scientist_agent()
analyst = aba.get_image_analyst()
reviewer = aba.get_paper_reviewer()

# Use agents directly
response = scientist.run("What is adaptive optics?")
```

#### Custom Router
```python
import aibioagent as aba

# Get router for custom workflows
router = aba.get_router()

# Route queries manually
response = router.route_query(
    query="Analyze this",
    image_path="image.tif",
    pdf_path=None
)
```

#### Package Information
```python
import aibioagent as aba

# Get configuration info
info = aba.info()
print(f"Database: {info['database_path']}")
print(f"LLM Model: {info['llm_model']}")
print(f"API Key Set: {info['api_key_set']}")
```

---

## 🎯 Common Use Cases

### Use Case 1: Build Custom Research Database

```python
import aibioagent as aba

# Setup
aba.set_api_key("sk-your-key")

# Add your research papers
aba.add_papers("papers/cell_biology", collection="cell_bio")
aba.add_papers("papers/imaging", collection="imaging")

# Add relevant documentation
aba.add_urls([
    "https://cellprofiler.org/",
    "https://napari.org/stable/"
], collection="tools")

# Query your custom database
response = aba.ask("What imaging tools are best for cell tracking?")
```

### Use Case 2: Paper Analysis Workflow

```python
import aibioagent as aba

aba.set_api_key("sk-your-key")

# Analyze multiple papers
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]

for paper in papers:
    print(f"\n{'='*60}")
    print(f"Analyzing: {paper}")
    print('='*60)
    
    # Get summary
    summary = aba.ask("Summarize key findings", pdf_path=paper)
    print(f"\nSummary:\n{summary}")
    
    # Get critique
    critique = aba.ask("What are limitations?", pdf_path=paper)
    print(f"\nLimitations:\n{critique}")
```

### Use Case 3: Image Analysis Pipeline

```python
import aibioagent as aba
from pathlib import Path

aba.set_api_key("sk-your-key")

# Analyze directory of images
image_dir = Path("microscopy_images")

for image_path in image_dir.glob("*.tif"):
    print(f"\nAnalyzing: {image_path.name}")
    
    # Get workflow suggestion
    workflow = aba.ask(
        "Suggest a complete analysis workflow",
        image_path=str(image_path)
    )
    
    print(f"Workflow:\n{workflow}\n")
```

### Use Case 4: Multi-Collection Research

```python
import aibioagent as aba

aba.set_api_key("sk-your-key")

# Build specialized collections
aba.add_papers("papers/crispr", collection="crispr")
aba.add_papers("papers/microscopy", collection="microscopy")
aba.add_papers("papers/segmentation", collection="segmentation")

# Search across collections
query = "CRISPR imaging techniques"

for collection in ["crispr", "microscopy", "segmentation"]:
    print(f"\n{'='*60}")
    print(f"Searching in: {collection}")
    print('='*60)
    
    results = aba.search_collection(query, collection, top_k=3)
```

### Use Case 5: Library Integration

```python
import aibioagent as aba

# Your existing analysis code
def analyze_image(image_path):
    # Your analysis logic here
    results = {"mean_intensity": 123.4, "cell_count": 42}
    return results

# Enhance with AI insights
def enhanced_analysis(image_path):
    # Run your analysis
    results = analyze_image(image_path)
    
    # Get AI suggestions
    ai_suggestion = aba.ask(
        f"I found {results['cell_count']} cells. "
        f"What additional analysis should I perform?",
        image_path=image_path
    )
    
    results["ai_suggestion"] = ai_suggestion
    return results

# Use it
report = enhanced_analysis("sample.tif")
print(report)
```

---

## 🔧 Configuration Tips

### Environment Variables

Create a `.env` file in your project:
```bash
OPENAI_API_KEY=sk-your-key-here
```

Or set via Python:
```python
import aibioagent as aba
aba.set_api_key("sk-your-key-here")
```

### Database Location

By default, databases are stored in `data/chroma_db/`. To check:
```python
import aibioagent as aba
info = aba.info()
print(f"Database: {info['database_path']}")
```

### Chunk Size Guidelines

- **Small chunks (500)**: Better for precise retrieval, more chunks
- **Medium chunks (1000)**: Balanced (default)
- **Large chunks (2000)**: More context, fewer chunks

```python
# For technical documentation (precise)
aba.add_papers("docs", chunk_size=500)

# For research papers (context)
aba.add_papers("papers", chunk_size=2000)
```

---

## 🐛 Troubleshooting

### API Key Issues
```python
# Check if key is set
import aibioagent as aba
key = aba.get_api_key()
if not key:
    print("API key not set!")
    aba.set_api_key("sk-your-key")
```

### Database Issues
```python
# List collections to verify
import aibioagent as aba
aba.list_collections()

# Search to test
results = aba.search_collection("test query", "your_collection")
```

### Empty Results
```python
# Make sure collection exists
collections = aba.list_collections()

# Make sure papers were added
aba.add_papers("papers", collection="test", verbose=True)
```

---

## 📚 Examples Repository

Check out complete examples:
- [examples/basic_usage.py](examples/basic_usage.py)
- [examples/paper_analysis.py](examples/paper_analysis.py)
- [examples/image_workflow.py](examples/image_workflow.py)
- [examples/custom_database.py](examples/custom_database.py)

---

## 🆘 Getting Help

```python
import aibioagent as aba

# Module help
help(aba)

# Function help
help(aba.add_papers)
help(aba.ask)

# Package info
aba.info()
```

For issues: https://github.com/chenli-git/AI_scientist_for_bioimaging/issues
