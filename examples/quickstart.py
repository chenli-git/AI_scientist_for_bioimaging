"""
Quick Start Example - Minimal Code
===================================
"""

import aibioagent as aba

# One-line setup
aba.quickstart(
    api_key="sk-your-key-here",
    pdf_folder="papers/"  # optional
)

# Ask questions
response = aba.ask("What is adaptive optics in microscopy?")
print(response)

# Analyze an image
response = aba.ask(
    "What segmentation method should I use?",
    image_path="microscopy_image.tif"
)
print(response)

# Review a paper
response = aba.ask(
    "Summarize this paper",
    pdf_path="research_paper.pdf"
)
print(response)
