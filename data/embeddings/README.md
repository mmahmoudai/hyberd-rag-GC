# Embeddings Directory

## Overview
This directory contains vector embeddings and indexes generated from the processed network security datasets. These embeddings are used by the RAG models for efficient retrieval and similarity matching.

## Directory Structure
- `/vector-embeddings/` - Vector representations of network security data
- `/indexes/` - FAISS and other index structures for efficient similarity search

## Embedding Types
The following types of embeddings are stored in this directory:

1. **Document Embeddings**: Vector representations of network packet chunks and flow records
2. **Entity Embeddings**: Vectors representing network entities (IPs, ports, protocols)
3. **Attack Pattern Embeddings**: Specialized vectors for known attack signatures
4. **Contextual Embeddings**: Embeddings that capture temporal and relational context

## Index Structures
For efficient similarity search, the following index structures are maintained:

1. **FAISS Indexes**: Optimized for high-dimensional vector search
2. **Graph Indexes**: For graph-based retrieval in Graph RAG
3. **Hybrid Indexes**: Specialized structures for the Hybrid Cache-Graph RAG

## Generation Process
Embeddings are generated using:
- Transformer-based models fine-tuned for network security
- Domain-specific encoding techniques
- Dimensionality reduction where appropriate
- Specialized embedding strategies for each RAG architecture

## Usage Guidelines
- Scripts for generating embeddings are in `/network-rag-research/src/data_processing/`
- Each RAG model has specific utilities for working with these embeddings
- Indexes should be rebuilt when new data is added to maintain performance
- For large-scale deployments, consider using incremental index updates

## Performance Considerations
- Vector dimensionality affects both accuracy and retrieval speed
- Index structure choice significantly impacts query performance
- Memory requirements scale with dataset size and embedding dimension
- Consider quantization for production deployments with memory constraints
