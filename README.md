# Knowledge Assistant with RAG Pipeline

✅ **An end-to-end Retrieval-Augmented Generation (RAG) system using open-source tools and models**

This project implements a **Knowledge Assistant** that answers user questions strictly based on the documents provided. The system demonstrates a full RAG pipeline, including document ingestion, embedding generation, vector search, prompt augmentation, and response generation, while ensuring grounded and reliable answers.

## Problem Statement

Build a system that:

- Answers questions based strictly on provided documents (PDF, TXT, Markdown)
- Implements a full RAG pipeline:
  1. Document ingestion
  2. Chunking & embedding
  3. Storing embeddings in a vector database
  4. User query embedding & vector search
  5. Context retrieval & prompt augmentation
  6. Response generation
- Uses open-source embeddings and LLMs
- Enforces safety controls and prevents hallucinations
- Includes references to source documents in responses
- Provides at least one interaction method (CLI, REST API, or UI)

## Features

- ✅ Ingest 5–20 documents (PDF, TXT, Markdown)  
- ✅ Chunk documents and generate embeddings using open-source models (Sentence Transformers, BGE, E5, Instructor, etc.)  
- ✅ Store embeddings in a vector database (FAISS, Chroma, Weaviate, Qdrant, Milvus, etc.)  
- ✅ Implement a complete RAG flow: query embedding → vector search → context retrieval → prompt augmentation → response generation  
- ✅ Prompt engineering with system instructions to prevent hallucinations  
- ✅ Basic guardrails:
  - Output filtering  
  - Confidence thresholding  
  - “Answer only from context” enforcement  
- ✅ Responses include source document references  
