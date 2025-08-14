# USC DPS Document Analysis – RAG System

This project is a **retrieval-augmented generation (RAG) system** built to help USC’s Department of Public Safety (DPS) work with their internal documents more effectively.  
DPS maintains hundreds of pages of safety protocols, incident reports, and policy manuals, which are often time-consuming to review manually.  

With this system, those documents are automatically ingested, chunked, and stored in a vector database for semantic search. Instead of digging through lengthy PDFs, staff can query the system in plain English and instantly retrieve the most relevant information. This reduces manual effort, speeds up the analysis process, and makes it easier to identify insights that support campus safety and incident response.  

---

## Features
- Converts PDFs into Markdown for easier processing  
- Splits large documents into smaller, queryable chunks  
- Embeds document chunks for semantic search  
- Natural language question-answering using an integrated LLM  
- Vector-based storage and retrieval for fast, accurate responses  
- Automated testing and evaluation with 100+ predefined cases  

---

## How It Works
1. **Document Ingestion** – PDFs are converted into Markdown using **Docling**  
2. **Chunking & Preprocessing** – Segmentation with **LangChain’s RecursiveCharacterTextSplitter**  
3. **Embedding & LLM Integration** – **Nomic/Mixedbread embeddings** + **deepseek-r1, etc. LLM via Ollama**  
4. **Vector Store** – Powered by **Chroma** for semantic search  
5. **Testing & Evaluation** – Accuracy validated with **pytest** and **Llama3, etc.**  

---

## Tech Stack
- **Docling** – PDF → Markdown conversion  
- **LangChain** – Document splitting & RAG pipeline  
- **Mixedbread** – Embedding model  
- **deepseek-r1 (via Ollama)** – Large Language Model for Q&A  
- **Chroma** – Vector database for storage & retrieval  
- **Pytest & Llama3** – Automated testing and evaluation  

---

## Getting Started

### Prerequisites
- Python 3.10+  
- [Ollama](https://ollama.ai/) installed locally  
- ChromaDB running for vector storage  
- Virtual environment recommended  

### Installation
```bash
git clone https://github.com/mk-8/Retrieval-Augmented-Generation_RAG.git
cd Retrieval-Augmented-Generation_RAG
pip install -r requirements.txt
