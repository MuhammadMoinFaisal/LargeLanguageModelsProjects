# LangChain & LLM Projects Collection

A suite of applications demonstrating document processing, chatbots, and AI-powered question answering using LangChain and various LLMs.

## 📚 Projects Overview

### 1. Book Summaries with LangChain and OpenAI
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.0.200-green)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt3.5-purple)

Automated book summarization system that processes PDFs/epubs and generates concise summaries using OpenAI's models.

**Features:**
- PDF text extraction
- Chunking and embeddings
- Summary generation with temperature control

---

### 2. Chat with Multiple Documents
![Streamlit](https://img.shields.io/badge/Streamlit-1.22-red)
![FAISS](https://img.shields.io/badge/FAISS-vector_store-yellow)

Multi-document QA system that allows conversational interaction with your document collection.

**Tech Stack:**
- Document loading (PDF, DOCX, TXT)
- Text splitting with RecursiveCharacterTextSplitter
- FAISS vector store for efficient similarity search

---

### 3. Website Chatbot
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-4.11-orange)
![Flask](https://img.shields.io/badge/Flask-2.2-lightgrey)

Custom chatbot that can be trained on any website content through web scraping.

**Implementation:**
- URL-based content ingestion
- Context-aware responses
- Conversation memory

---

### 4. LangChain Crash Course
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

Educational materials covering LangChain fundamentals through practical examples.

**Topics Include:**
- Chains, Agents, and Memory
- Document Loaders and Text Splitters
- Vector Stores and Retrievers

---

### 5. Llama2 Integration Projects
![Llama2](https://img.shields.io/badge/Meta-Llama2-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

Collection of implementations using Meta's Llama2 models:

#### a. QA Book PDF with LangChain
- PDF question answering system
- Custom prompt templates
- Retrieval-Augmented Generation (RAG)

#### b. Run Llama2 on Google Colab
- Free-tier Colab implementation
- Quantized model support
- Gradio interface

#### c. Local CPU Deployment
- Optimized for CPU-only environments
- Quantized model loading
- Basic chat interface

## 🛠 Installation

```bash
git clone https://github.com/yourusername/llm-projects.git
cd llm-projects
pip install -r requirements.txt
