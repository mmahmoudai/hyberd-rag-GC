"""
Implementation of the Traditional RAG system for network security packet analysis.
"""

import os
import json
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from ..common.base import Document, Query, RetrievalResult, EmbeddingModel, Retriever, Generator, RAGSystem
from ..common.config import VECTOR_DB_CONFIG, EMBEDDING_CONFIG
from ..utils.network_utils import format_packet_data, generate_flow_id, extract_protocol_features

# Configure logging
logger = logging.getLogger(__name__)

class NetworkEmbeddingModel(EmbeddingModel):
    """
    Embedding model for network security data.
    Uses a sentence transformer model optimized for technical text.
    """
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name or EMBEDDING_CONFIG["model_name"]
        self.batch_size = EMBEDDING_CONFIG["batch_size"]
        self.max_seq_length = EMBEDDING_CONFIG["max_seq_length"]
        self.normalize = EMBEDDING_CONFIG["normalize_embeddings"]
        
        # Defer actual model loading to when it's needed to save memory
        self._model = None
    
    def _load_model(self):
        """Load the model if not already loaded."""
        if self._model is None:
            try:
                # Import here to avoid loading dependencies until needed
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                # Fallback to a simple embedding method if sentence_transformers is not available
                logger.warning("SentenceTransformer not available. Using simple fallback embedding method.")
                self._model = "fallback"
    
    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """
        Simple fallback encoding method when sentence_transformers is not available.
        Uses word frequency and character n-grams for a basic embedding.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create a TF-IDF vectorizer with a fixed output dimension
        vectorizer = TfidfVectorizer(
            max_features=min(1536, len(texts) * 10),  # Limit features
            ngram_range=(1, 3)  # Use unigrams, bigrams, and trigrams
        )
        
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        
        # Pad or truncate to match the expected dimension
        target_dim = VECTOR_DB_CONFIG["dimension"]
        result = np.zeros((len(texts), target_dim))
        
        for i, embedding in enumerate(tfidf_matrix):
            if embedding.shape[0] >= target_dim:
                result[i] = embedding[:target_dim]
            else:
                result[i, :embedding.shape[0]] = embedding
        
        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            result = result / norms
        
        return result
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        self._load_model()
        
        if self._model == "fallback":
            return self._fallback_encode(texts)
        
        # Use the sentence transformer model
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
    
    def encode_documents(self, documents: List[Document]) -> List[Document]:
        """
        Encode documents and update their embedding attribute.
        
        Args:
            documents: List of documents to encode
            
        Returns:
            List of documents with embeddings
        """
        texts = [doc.content for doc in documents]
        embeddings = self.encode(texts)
        
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
        
        return documents
    
    def encode_query(self, query: Query) -> Query:
        """
        Encode query and update its embedding attribute.
        
        Args:
            query: Query to encode
            
        Returns:
            Query with embedding
        """
        embedding = self.encode([query.text])[0]
        query.embedding = embedding
        return query


class VectorRetriever(Retriever):
    """
    Vector-based retriever for network security data.
    Uses FAISS for efficient similarity search.
    """
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        index_type: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None
    ):
        """
        Initialize the vector retriever.
        
        Args:
            embedding_model: Model for encoding documents and queries
            index_type: Type of FAISS index to use
            dimension: Dimension of the embeddings
            metric: Distance metric to use
        """
        self.embedding_model = embedding_model or NetworkEmbeddingModel()
        self.index_type = index_type or VECTOR_DB_CONFIG["index_type"]
        self.dimension = dimension or VECTOR_DB_CONFIG["dimension"]
        self.metric = metric or VECTOR_DB_CONFIG["metric"]
        
        # Initialize FAISS index
        self._init_index()
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        self.doc_ids: List[str] = []
    
    def _init_index(self):
        """Initialize the FAISS index based on configuration."""
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Use more advanced index types if specified
        if self.index_type == "HNSW":
            hnsw_index = faiss.IndexHNSWFlat(
                self.dimension, 
                VECTOR_DB_CONFIG["ef_construction"]
            )
            hnsw_index.hnsw.efSearch = VECTOR_DB_CONFIG["ef_search"]
            self.index = hnsw_index
        elif self.index_type == "IVF":
            # For IVF, we need some data to train the index
            # We'll initialize with a basic index and convert later when data is available
            self.convert_to_ivf = True
        else:
            self.convert_to_ivf = False
    
    def _maybe_convert_to_ivf(self, embeddings: np.ndarray):
        """
        Convert to IVF index if specified and enough data is available.
        
        Args:
            embeddings: Embeddings to use for training
        """
        if self.convert_to_ivf and len(self.documents) >= 1000:
            n_clusters = min(int(np.sqrt(len(self.documents))), 256)
            
            if self.metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.dimension)
                ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
            
            ivf_index.train(embeddings)
            ivf_index.nprobe = VECTOR_DB_CONFIG["nprobe"]
            
            # Copy data from old index to new index
            if self.index.ntotal > 0:
                all_embeddings = np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)])
                ivf_index.add(all_embeddings)
            
            self.index = ivf_index
            self.convert_to_ivf = False
            logger.info(f"Converted to IVF index with {n_clusters} clusters")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
        
        # Encode documents if not already encoded
        docs_to_encode = [doc for doc in documents if doc.embedding is None]
        if docs_to_encode:
            self.embedding_model.encode_documents(docs_to_encode)
        
        # Extract embeddings
        embeddings = np.vstack([doc.embedding for doc in documents])
        
        # Maybe convert to IVF index
        self._maybe_convert_to_ivf(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents
        for doc in documents:
            self.documents[doc.id] = doc
            self.doc_ids.append(doc.id)
        
        logger.info(f"Added {len(documents)} documents to retriever. Total: {len(self.documents)}")
    
    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query to retrieve documents for
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieval result containing documents and scores
        """
        # Encode query if not already encoded
        if query.embedding is None:
            query = self.embedding_model.encode_query(query)
        
        # Reshape for FAISS
        query_embedding = query.embedding.reshape(1, -1)
        
        # Search
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return RetrievalResult([], [], {"message": "No documents in index"})
        
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        # Get documents
        retrieved_docs = []
        retrieved_scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                if doc_id in self.documents:
                    retrieved_docs.append(self.documents[doc_id])
                    retrieved_scores.append(float(scores[0][i]))
        
        # Apply filters if specified
        if query.filters:
            filtered_docs = []
            filtered_scores = []
            
            for doc, score in zip(retrieved_docs, retrieved_scores):
                if self._matches_filters(doc, query.filters):
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
            
            retrieved_docs = filtered_docs
            retrieved_scores = filtered_scores
        
        return RetrievalResult(
            documents=retrieved_docs,
            scores=retrieved_scores,
            metadata={"query": query.text}
        )
    
    def _matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """
        Check if a document matches the specified filters.
        
        Args:
            doc: Document to check
            filters: Filters to apply
            
        Returns:
            True if document matches filters, False otherwise
        """
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            
            if isinstance(value, list):
                if doc.metadata[key] not in value:
                    return False
            elif doc.metadata[key] != value:
                return False
        
        return True
    
    def update_documents(self, documents: List[Document]) -> None:
        """
        Update existing documents in the retriever.
        
        Args:
            documents: List of documents to update
        """
        # For simplicity, we'll just delete and re-add the documents
        doc_ids = [doc.id for doc in documents]
        self.delete_documents(doc_ids)
        self.add_documents(documents)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the retriever.
        
        Args:
            document_ids: List of document IDs to delete
        """
        # FAISS doesn't support direct deletion, so we need to rebuild the index
        remaining_docs = []
        
        for doc_id, doc in self.documents.items():
            if doc_id not in document_ids:
                remaining_docs.append(doc)
        
        # Clear existing data
        self.documents = {}
        self.doc_ids = []
        
        # Reinitialize index
        self._init_index()
        
        # Re-add remaining documents
        if remaining_docs:
            self.add_documents(remaining_docs)
        
        logger.info(f"Deleted {len(document_ids)} documents from retriever. Remaining: {len(self.documents)}")


class NetworkSecurityGenerator(Generator):
    """
    Generator for network security analysis.
    """
    def __init__(self):
        """Initialize the generator."""
        pass
    
    def generate(
        self, 
        query: Query, 
        retrieval_result: RetrievalResult
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: The original query
            retrieval_result: Result from the retriever
            
        Returns:
            Generated response and metadata
        """
        # For now, implement a simple rule-based generator
        # In a full implementation, this would use an LLM
        
        if not retrieval_result.documents:
            return {
                "response": "No relevant network traffic found for this query.",
                "confidence": 0.0,
                "sources": []
            }
        
        # Extract relevant information from retrieved documents
        sources = []
        protocols = set()
        ips = set()
        ports = set()
        potential_threats = []
        
        for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
            sources.append({
                "id": doc.id,
                "relevance": score,
                "metadata": doc.metadata
            })
            
            # Extract key information
            if 'protocol' in doc.metadata:
                protocols.add(doc.metadata['protocol'])
            
            if 'src_ip' in doc.metadata:
                ips.add(doc.metadata['src_ip'])
            if 'dst_ip' in doc.metadata:
                ips.add(doc.metadata['dst_ip'])
            
            if 'src_port' in doc.metadata:
                ports.add(doc.metadata['src_port'])
            if 'dst_port' in doc.metadata:
                ports.add(doc.metadata['dst_port'])
            
            # Check for threat indicators
            if 'threat_indicators' in doc.metadata and doc.metadata['threat_indicators']:
                potential_threats.extend(doc.metadata['threat_indicators'])
        
        # Generate a simple response
        response_parts = []
        
        # Query understanding
        response_parts.append(f"Query: {query.text}")
        
        # Summary of findings
        response_parts.append(f"\nAnalysis based on {len(retrieval_result.documents)} relevant network flows:")
        
        if protocols:
            response_parts.append(f"- Protocols: {', '.join(protocols)}")
        
        if ips:
            response_parts.append(f"- IP addresses: {', '.join(list(ips)[:5])}" + 
                                 (f" and {len(ips)-5} more" if len(ips) > 5 else ""))
        
        if ports:
            response_parts.append(f"- Ports: {', '.join(map(str, list(ports)[:5]))}" +
                                 (f" and {len(ports)-5} more" if len(ports) > 5 else ""))
        
        # Threat assessment
        if potential_threats:
            response_parts.append("\nPotential security concerns:")
            for threat in set(potential_threats):
                response_parts.append(f"- {threat}")
            
            confidence = min(0.5 + (len(potential_threats) / 10), 0.9)
        else:
            response_parts.append("\nNo immediate security concerns identified in the analyzed traffic.")
            confidence = 0.7
        
        # Recommendations
        response_parts.append("\nRecommendations:")
        if potential_threats:
            response_parts.append("- Further investigation recommended for the identified potential threats")
            response_parts.append("- Consider additional monitoring for the involved IP addresses and ports")
        else:
            response_parts.append("- Continue monitoring for any changes in network behavior")
            response_parts.append("- Regular security audits recommended as part of standard practice")
        
        return {
            "response": "\n".join(response_parts),
            "confidence": confidence,
            "sources": sources,
            "protocols": list(protocols),
            "ip_addresses": list(ips),
            "ports": list(ports),
            "potential_threats": list(set(potential_threats))
        }


class TraditionalRAG(RAGSystem):
    """
    Traditional RAG system for network security packet analysis.
    """
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None
    ):
        """
        Initialize the Traditional RAG system.
        
        Args:
            embedding_model: Model for encoding documents and queries
            retriever: Retriever component
            generator: Generator component
        """
        self.embedding_model = embedding_model or NetworkEmbeddingModel()
        self.retriever = retriever or VectorRetriever(embedding_model=self.embedding_model)
        self.generator = generator or NetworkSecurityGenerator()
    
    def process_query(
        self, 
        query: Union[str, Query], 
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG system.
        
        Args:
            query: Query text or Query object
            top_k: Number of documents to retrieve
            
        Returns:
            Generated response and metadata
        """
        # Convert string to Query object if needed
        if isinstance(query, str):
            query = Query(text=query)
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(query, top_k=top_k)
        
        # Generate response
        response = self.generator.generate(query, retrieval_result)
        
        return response
    
    def add_documents(self, documents: List[Union[Dict[str, Any], Document]]) -> None:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of documents to add
        """
        # Convert dictionaries to Document objects if needed
        doc_objects = []
        
        for doc in documents:
            if isinstance(doc, dict):
                # Generate ID if not provided
                if 'id' not in doc:
                    doc['id'] = generate_flow_id(doc)
                
                # Format content if not provided
                if 'content' not in doc:
                    doc['content'] = format_packet_data(doc)
                
                # Extract protocol features if not in metadata
                metadata = doc.get('metadata', {})
                if not metadata:
                    metadata = {k: v for k, v in doc.items() if k not in ['id', 'content']}
                    protocol_features = extract_protocol_features(doc)
                    metadata.update(protocol_features)
                
                doc_obj = Document(
                    id=doc['id'],
                    content=doc['content'],
                    metadata=metadata
                )
            else:
                doc_obj = doc
            
            doc_objects.append(doc_obj)
        
        # Add to retriever
        self.retriever.add_documents(doc_objects)
    
    def evaluate(
        self, 
        queries: List[Query], 
        ground_truth: List[Any]
    ) -> Dict[str, float]:
        """
        Evaluate the RAG system.
        
        Args:
            queries: List of queries to evaluate
            ground_truth: Ground truth for each query
            
        Returns:
            Evaluation metrics
        """
        # Simple evaluation implementation
        results = []
        
        for query in queries:
            result = self.process_query(query)
            results.append(result)
        
        # Calculate metrics
        # This is a placeholder - actual metrics would depend on the specific evaluation task
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "latency": 0.0
        }
        
        return metrics
