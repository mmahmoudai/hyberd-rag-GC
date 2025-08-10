"""
Implementation of the Hybrid Cache-Graph RAG system for network security packet analysis.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import networkx as nx

from ..common.base import Document, Query, RetrievalResult, EmbeddingModel, Retriever, Generator, RAGSystem
from ..common.config import HYBRID_CONFIG
from ..traditional_rag.traditional_rag import NetworkEmbeddingModel, NetworkSecurityGenerator
from ..graph_rag.graph_rag import NetworkGraph, GraphBuilder, NetworkNode, NetworkEdge
from ..cache_rag.cache_rag import CacheSystem, CacheKey, CacheEntry
from ..utils.network_utils import format_packet_data, generate_flow_id, extract_protocol_features

# Configure logging
logger = logging.getLogger(__name__)

class HybridRetriever(Retriever):
    """
    Hybrid retriever that combines graph-based and cached retrieval.
    """
    def __init__(
        self,
        graph: Optional[NetworkGraph] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        cache_system: Optional[CacheSystem] = None,
        cache_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        adaptive: Optional[bool] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            graph: Network graph
            embedding_model: Model for encoding queries
            cache_system: Cache system
            cache_weight: Weight for cache component
            graph_weight: Weight for graph component
            adaptive: Whether to use adaptive weighting
        """
        self.graph = graph or NetworkGraph()
        self.embedding_model = embedding_model or NetworkEmbeddingModel()
        self.cache_system = cache_system or CacheSystem()
        self.graph_builder = GraphBuilder(self.graph)
        
        # Weighting parameters
        self.cache_weight = cache_weight if cache_weight is not None else HYBRID_CONFIG["cache_weight"]
        self.graph_weight = graph_weight if graph_weight is not None else HYBRID_CONFIG["graph_weight"]
        self.adaptive = adaptive if adaptive is not None else HYBRID_CONFIG["adaptive"]
        self.synergy_threshold = HYBRID_CONFIG["synergy_threshold"]
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "cache_hits": 0,
            "graph_hits": 0,
            "hybrid_hits": 0,
            "total_queries": 0,
            "cache_latency": [],
            "graph_latency": [],
            "hybrid_latency": []
        }
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
        
        # Store documents
        for doc in documents:
            self.documents[doc.id] = doc
        
        # Build graph
        self.graph_builder.build_from_documents(documents)
        
        # Invalidate cache entries that might be affected
        self._invalidate_affected_cache_entries(documents)
        
        logger.info(f"Added {len(documents)} documents to hybrid retriever. Total: {len(self.documents)}")
    
    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query to retrieve documents for
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieval result containing documents and scores
        """
        self.performance_metrics["total_queries"] += 1
        
        # Generate cache key
        cache_key = CacheKey.generate_key(query, {"top_k": top_k})
        
        # Check cache first
        cache_start_time = time.time()
        cached_result = self.cache_system.get(cache_key)
        cache_latency = time.time() - cache_start_time
        self.performance_metrics["cache_latency"].append(cache_latency)
        
        if cached_result is not None:
            logger.info(f"Cache hit for query: {query.text}")
            self.performance_metrics["cache_hits"] += 1
            return cached_result
        
        # Cache miss - try graph-based retrieval
        graph_start_time = time.time()
        graph_result = self._graph_retrieve(query, top_k)
        graph_latency = time.time() - graph_start_time
        self.performance_metrics["graph_latency"].append(graph_latency)
        
        if graph_result.documents:
            logger.info(f"Graph hit for query: {query.text}")
            self.performance_metrics["graph_hits"] += 1
            
            # Cache the result
            self.cache_system.put(
                key=cache_key,
                value=graph_result,
                metadata={"query": query.text, "top_k": top_k, "source": "graph"}
            )
            
            return graph_result
        
        # If both cache and graph miss, use hybrid approach
        hybrid_start_time = time.time()
        hybrid_result = self._hybrid_retrieve(query, top_k)
        hybrid_latency = time.time() - hybrid_start_time
        self.performance_metrics["hybrid_latency"].append(hybrid_latency)
        
        logger.info(f"Hybrid retrieval for query: {query.text}")
        self.performance_metrics["hybrid_hits"] += 1
        
        # Cache the result
        self.cache_system.put(
            key=cache_key,
            value=hybrid_result,
            metadata={"query": query.text, "top_k": top_k, "source": "hybrid"}
        )
        
        return hybrid_result
    
    def _graph_retrieve(self, query: Query, top_k: int) -> RetrievalResult:
        """
        Retrieve documents using graph-based approach.
        
        Args:
            query: Query to retrieve documents for
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieval result containing documents and scores
        """
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        
        if not query_entities:
            logger.warning("No entities extracted from query")
            return RetrievalResult([], [], {"message": "No entities extracted from query"})
        
        # Find relevant nodes
        relevant_nodes = self._find_relevant_nodes(query_entities)
        
        if not relevant_nodes:
            logger.warning("No relevant nodes found")
            return RetrievalResult([], [], {"message": "No relevant nodes found"})
        
        # Get documents associated with the nodes
        document_scores = self._score_documents_by_nodes(relevant_nodes)
        
        # Sort by score and take top k
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]
        
        # Get document objects
        retrieved_docs = []
        retrieved_scores = []
        
        for doc_id, score in top_docs:
            if doc_id in self.documents:
                retrieved_docs.append(self.documents[doc_id])
                retrieved_scores.append(score)
        
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
            metadata={
                "query": query.text,
                "entities": query_entities,
                "nodes": relevant_nodes,
                "source": "graph"
            }
        )
    
    def _hybrid_retrieve(self, query: Query, top_k: int) -> RetrievalResult:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query to retrieve documents for
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieval result containing documents and scores
        """
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        
        # Find relevant nodes (even if no entities were extracted)
        relevant_nodes = self._find_relevant_nodes(query_entities)
        
        # Get graph-based document scores
        graph_document_scores = self._score_documents_by_nodes(relevant_nodes)
        
        # Get text-based document scores
        # For this, we need to encode the query if not already encoded
        if query.embedding is None:
            query = self.embedding_model.encode_query(query)
        
        # Calculate text similarity for all documents
        text_document_scores = {}
        
        for doc_id, doc in self.documents.items():
            # Encode document if not already encoded
            if doc.embedding is None:
                doc = self.embedding_model.encode_documents([doc])[0]
            
            # Calculate similarity
            if doc.embedding is not None and query.embedding is not None:
                similarity = np.dot(query.embedding, doc.embedding) / (
                    np.linalg.norm(query.embedding) * np.linalg.norm(doc.embedding)
                )
                text_document_scores[doc_id] = float(similarity)
        
        # Combine scores using adaptive weighting if enabled
        if self.adaptive:
            # Calculate synergy factor
            synergy_factor = self._calculate_synergy(graph_document_scores, text_document_scores)
            
            # Adjust weights based on synergy
            if synergy_factor > self.synergy_threshold:
                # High synergy - boost graph component
                adjusted_graph_weight = self.graph_weight * (1 + (synergy_factor - self.synergy_threshold))
                adjusted_cache_weight = 1 - adjusted_graph_weight
            else:
                # Low synergy - use default weights
                adjusted_graph_weight = self.graph_weight
                adjusted_cache_weight = self.cache_weight
        else:
            # Use fixed weights
            adjusted_graph_weight = self.graph_weight
            adjusted_cache_weight = self.cache_weight
        
        # Combine scores
        combined_scores = {}
        
        # Get all document IDs from both score sets
        all_doc_ids = set(graph_document_scores.keys()) | set(text_document_scores.keys())
        
        for doc_id in all_doc_ids:
            graph_score = graph_document_scores.get(doc_id, 0.0)
            text_score = text_document_scores.get(doc_id, 0.0)
            
            combined_scores[doc_id] = (
                adjusted_graph_weight * graph_score +
                adjusted_cache_weight * text_score
            )
        
        # Sort by score and take top k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]
        
        # Get document objects
        retrieved_docs = []
        retrieved_scores = []
        
        for doc_id, score in top_docs:
            if doc_id in self.documents:
                retrieved_docs.append(self.documents[doc_id])
                retrieved_scores.append(score)
        
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
            metadata={
                "query": query.text,
                "entities": query_entities,
                "nodes": relevant_nodes,
                "source": "hybrid",
                "graph_weight": adjusted_graph_weight,
                "text_weight": adjusted_cache_weight
            }
        )
    
    def _calculate_synergy(
        self, 
        graph_scores: Dict[str, float], 
        text_scores: Dict[str, float]
    ) -> float:
        """
        Calculate synergy factor between graph and text scores.
        
        Args:
            graph_scores: Document scores from graph-based retrieval
            text_scores: Document scores from text-based retrieval
            
        Returns:
            Synergy factor (0-1)
        """
        # Get top documents from each method
        top_k = 10
        
        top_graph_docs = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_text_docs = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Convert to sets for overlap calculation
        graph_doc_ids = {doc_id for doc_id, _ in top_graph_docs}
        text_doc_ids = {doc_id for doc_id, _ in top_text_docs}
        
        # Calculate overlap
        overlap = len(graph_doc_ids & text_doc_ids)
        
        # Calculate synergy factor
        if not graph_doc_ids or not text_doc_ids:
            return 0.0
        
        # Jaccard similarity
        synergy = overlap / len(graph_doc_ids | text_doc_ids)
        
        return synergy
    
    def _extract_entities_from_query(self, query: Query) -> Dict[str, List[str]]:
        """
        Extract entities from a query.
        
        Args:
            query: Query to extract entities from
            
        Returns:
            Dictionary mapping entity types to lists of values
        """
        entities = {
            "ip_address": [],
            "port": [],
            "protocol": []
        }
        
        # Simple pattern matching for common entities
        
        # IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ip_matches = re.findall(ip_pattern, query.text)
        entities["ip_address"].extend(ip_matches)
        
        # Ports
        port_pattern = r'\b(?:port\s+|:)(\d+)\b'
        port_matches = re.findall(port_pattern, query.text)
        entities["port"].extend(port_matches)
        
        # Protocols
        protocol_keywords = ["http", "https", "dns", "tcp", "udp", "icmp", "ssh", "ftp", "smtp", "telnet"]
        for protocol in protocol_keywords:
            if re.search(r'\b' + protocol + r'\b', query.text.lower()):
                entities["protocol"].append(protocol)
        
        return entities
    
    def _find_relevant_nodes(self, query_entities: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        """
        Find nodes relevant to the query entities.
        
        Args:
            query_entities: Dictionary mapping entity types to lists of values
            
        Returns:
            List of (node_id, relevance_score) tuples
        """
        relevant_nodes = []
        
        # Find exact matches
        for entity_type, values in query_entities.items():
            for value in values:
                if entity_type == "ip_address":
                    node_id = f"ip:{value}"
                    if node_id in self.graph.nodes:
                        relevant_nodes.append((node_id, 1.0))
                
                elif entity_type == "port":
                    node_id = f"port:{value}"
                    if node_id in self.graph.nodes:
                        relevant_nodes.append((node_id, 1.0))
                
                elif entity_type == "protocol":
                    node_id = f"protocol:{value}"
                    if node_id in self.graph.nodes:
                        relevant_nodes.append((node_id, 1.0))
        
        # If no exact matches, try to find similar nodes
        if not relevant_nodes:
            # This would be more sophisticated in a full implementation
            pass
        
        # Expand to neighbors with lower scores
        expanded_nodes = []
        
        for node_id, score in relevant_nodes:
            neighbors = self.graph.get_neighbors(node_id)
            for neighbor in neighbors:
                expanded_nodes.append((neighbor, score * 0.8))
        
        relevant_nodes.extend(expanded_nodes)
        
        # Deduplicate and normalize scores
        node_scores = {}
        for node_id, score in relevant_nodes:
            if node_id in node_scores:
                node_scores[node_id] = max(node_scores[node_id], score)
            else:
                node_scores[node_id] = score
        
        return [(node_id, score) for node_id, score in node_scores.items()]
    
    def _score_documents_by_nodes(self, relevant_nodes: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Score documents based on their association with relevant nodes.
        
        Args:
            relevant_nodes: List of (node_id, relevance_score) tuples
            
        Returns:
            Dictionary mapping document IDs to relevance scores
        """
        document_scores = {}
        
        for node_id, node_score in relevant_nodes:
            # Get documents associated with this node
            doc_ids = self.graph.get_documents_for_node(node_id)
            
            for doc_id in doc_ids:
                # Adjust score based on node centrality
                centrality = self.graph.compute_node_centrality(node_id)
                adjusted_score = node_score * (1 + centrality)
                
                if doc_id in document_scores:
                    document_scores[doc_id] += adjusted_score
                else:
                    document_scores[doc_id] = adjusted_score
        
        # Normalize scores
        if document_scores:
            max_score = max(document_scores.values())
            if max_score > 0:
                for doc_id in document_scores:
                    document_scores[doc_id] /= max_score
        
        return document_scores
    
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
    
    def _invalidate_affected_cache_entries(self, documents: List[Document]) -> None:
        """
        Invalidate cache entries that might be affected by document changes.
        
        Args:
            documents: List of documents that were added or updated
        """
        # This is a simplified approach - a more sophisticated implementation
        # would be more selective about which cache entries to invalidate
        
        # Extract key terms from documents
        terms = set()
        for doc in documents:
            # Extract terms from content
            doc_terms = doc.content.lower().split()
            terms.update(doc_terms)
            
            # Extract terms from metadata
            for key, value in doc.metadata.items():
                if isinstance(value, str):
                    terms.add(value.lower())
                elif isinstance(value, (int, float)):
                    terms.add(str(value))
        
        # Invalidate cache entries containing these terms
        # This is a very aggressive approach and would be refined in a real implementation
        self.cache_system.clear()
    
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
        # Remove from document storage
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
        
        # Remove from graph
        # This is a simplified implementation - a full implementation would
        # need to handle removing nodes and edges properly
        for doc_id in document_ids:
            if doc_id in self.graph.document_to_subgraph:
                del self.graph.document_to_subgraph[doc_id]
        
        # Invalidate cache entries
        self.cache_system.clear()
        
        logger.info(f"Deleted {len(document_ids)} documents from hybrid retriever. Remaining: {len(self.documents)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.performance_metrics.copy()
        
        # Calculate averages
        if metrics["cache_latency"]:
            metrics["avg_cache_latency"] = sum(metrics["cache_latency"]) / len(metrics["cache_latency"])
        else:
            metrics["avg_cache_latency"] = 0.0
        
        if metrics["graph_latency"]:
            metrics["avg_graph_latency"] = sum(metrics["graph_latency"]) / len(metrics["graph_latency"])
        else:
            metrics["avg_graph_latency"] = 0.0
        
        if metrics["hybrid_latency"]:
            metrics["avg_hybrid_latency"] = sum(metrics["hybrid_latency"]) / len(metrics["hybrid_latency"])
        else:
            metrics["avg_hybrid_latency"] = 0.0
        
        # Calculate hit rates
        if metrics["total_queries"] > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["total_queries"]
            metrics["graph_hit_rate"] = metrics["graph_hits"] / metrics["total_queries"]
            metrics["hybrid_hit_rate"] = metrics["hybrid_hits"] / metrics["total_queries"]
        else:
            metrics["cache_hit_rate"] = 0.0
            metrics["graph_hit_rate"] = 0.0
            metrics["hybrid_hit_rate"] = 0.0
        
        return metrics


class HybridCacheGraphRAG(RAGSystem):
    """
    Hybrid Cache-Graph RAG system for network security packet analysis.
    """
    def __init__(
        self,
        graph: Optional[NetworkGraph] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        cache_system: Optional[CacheSystem] = None,
        generator: Optional[Generator] = None,
        cache_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        adaptive: Optional[bool] = None
    ):
        """
        Initialize the Hybrid Cache-Graph RAG system.
        
        Args:
            graph: Network graph
            embedding_model: Model for encoding documents and queries
            cache_system: Cache system
            generator: Generator component
            cache_weight: Weight for cache component
            graph_weight: Weight for graph component
            adaptive: Whether to use adaptive weighting
        """
        self.embedding_model = embedding_model or NetworkEmbeddingModel()
        self.graph = graph or NetworkGraph()
        self.cache_system = cache_system or CacheSystem()
        self.generator = generator or NetworkSecurityGenerator()
        
        # Set up hybrid retriever
        self.retriever = HybridRetriever(
            graph=self.graph,
            embedding_model=self.embedding_model,
            cache_system=self.cache_system,
            cache_weight=cache_weight,
            graph_weight=graph_weight,
            adaptive=adaptive
        )
    
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
        
        # Check if we have a cached final result
        cache_key = CacheKey.generate_key(query, {"top_k": top_k, "final": True})
        cached_result = self.cache_system.get(cache_key)
        
        if cached_result is not None:
            logger.info(f"Final result cache hit for query: {query.text}")
            return cached_result
        
        # No final result cache hit, proceed with retrieval and generation
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(query, top_k=top_k)
        
        # Generate response
        response = self.generator.generate(query, retrieval_result)
        
        # Add performance information
        performance_metrics = self.retriever.get_performance_metrics()
        
        response["performance"] = {
            "processing_time": time.time() - start_time,
            "retrieval_stats": {
                "cache_hit_rate": performance_metrics["cache_hit_rate"],
                "graph_hit_rate": performance_metrics["graph_hit_rate"],
                "hybrid_hit_rate": performance_metrics["hybrid_hit_rate"],
                "avg_cache_latency": performance_metrics["avg_cache_latency"],
                "avg_graph_latency": performance_metrics["avg_graph_latency"],
                "avg_hybrid_latency": performance_metrics["avg_hybrid_latency"]
            }
        }
        
        # Add graph information if available
        if "entities" in retrieval_result.metadata and "nodes" in retrieval_result.metadata:
            response["graph_context"] = {
                "entities": retrieval_result.metadata["entities"],
                "nodes": retrieval_result.metadata["nodes"]
            }
        
        # Cache the final result
        self.cache_system.put(
            key=cache_key,
            value=response,
            metadata={"query": query.text, "top_k": top_k, "final": True}
        )
        
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
        processing_times = []
        
        for query in queries:
            start_time = time.time()
            result = self.process_query(query)
            processing_time = time.time() - start_time
            
            results.append(result)
            processing_times.append(processing_time)
        
        # Get performance metrics from retriever
        retriever_metrics = self.retriever.get_performance_metrics()
        
        # Calculate metrics
        # This is a placeholder - actual metrics would depend on the specific evaluation task
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "latency": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "cache_hit_rate": retriever_metrics["cache_hit_rate"],
            "graph_hit_rate": retriever_metrics["graph_hit_rate"],
            "hybrid_hit_rate": retriever_metrics["hybrid_hit_rate"]
        }
        
        return metrics
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.cache_system.clear()
        logger.info("All caches cleared")
