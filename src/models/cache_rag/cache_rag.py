"""
Implementation of the Cache RAG system for network security packet analysis.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import redis
import pickle
import hashlib

from ..common.base import Document, Query, RetrievalResult, EmbeddingModel, Retriever, Generator, RAGSystem
from ..common.config import CACHE_CONFIG
from ..traditional_rag.traditional_rag import NetworkEmbeddingModel, VectorRetriever, NetworkSecurityGenerator
from ..utils.network_utils import format_packet_data, generate_flow_id, extract_protocol_features

# Configure logging
logger = logging.getLogger(__name__)

class CacheKey:
    """
    Class for generating and managing cache keys.
    """
    @staticmethod
    def generate_key(query: Query, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: Query to generate key for
            context: Optional context information
            
        Returns:
            Cache key string
        """
        # Start with the query text
        key_components = [query.text]
        
        # Add filters if present
        if query.filters:
            filter_str = json.dumps(query.filters, sort_keys=True)
            key_components.append(filter_str)
        
        # Add context if present
        if context:
            context_str = json.dumps(context, sort_keys=True)
            key_components.append(context_str)
        
        # Create a hash of the components
        key_string = "||".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def generate_partial_key(query_terms: List[str]) -> str:
        """
        Generate a partial key for query terms.
        
        Args:
            query_terms: List of query terms
            
        Returns:
            Partial cache key string
        """
        # Sort terms for consistency
        sorted_terms = sorted(query_terms)
        key_string = "||".join(sorted_terms)
        return hashlib.md5(key_string.encode()).hexdigest()


class CacheEntry:
    """
    Class representing a cache entry.
    """
    def __init__(
        self,
        key: str,
        value: Any,
        timestamp: float,
        ttl: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            timestamp: Creation timestamp
            ttl: Time-to-live in seconds
            metadata: Optional metadata
        """
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.ttl = ttl
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_access = timestamp
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """
        Check if the entry is expired.
        
        Args:
            current_time: Current time (defaults to time.time())
            
        Returns:
            True if expired, False otherwise
        """
        if current_time is None:
            current_time = time.time()
        
        return current_time > (self.timestamp + self.ttl)
    
    def access(self) -> None:
        """Record an access to this cache entry."""
        self.access_count += 1
        self.last_access = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access": self.last_access
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create entry from dictionary representation."""
        entry = cls(
            key=data["key"],
            value=data["value"],
            timestamp=data["timestamp"],
            ttl=data["ttl"],
            metadata=data["metadata"]
        )
        entry.access_count = data["access_count"]
        entry.last_access = data["last_access"]
        return entry


class CacheSystem:
    """
    Cache system for RAG queries and results.
    """
    def __init__(
        self,
        max_size: Optional[int] = None,
        ttl: Optional[int] = None,
        policy: Optional[str] = None,
        redis_url: Optional[str] = None
    ):
        """
        Initialize the cache system.
        
        Args:
            max_size: Maximum number of entries
            ttl: Default time-to-live in seconds
            policy: Cache replacement policy
            redis_url: Optional Redis URL for distributed cache
        """
        self.max_size = max_size or CACHE_CONFIG["max_size"]
        self.ttl = ttl or CACHE_CONFIG["ttl"]
        self.policy = policy or CACHE_CONFIG["policy"]
        
        # Initialize Redis client if URL provided
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        # Try Redis first if available
        if self.redis_client:
            try:
                redis_value = self.redis_client.get(key)
                if redis_value:
                    entry_dict = pickle.loads(redis_value)
                    entry = CacheEntry.from_dict(entry_dict)
                    
                    if entry.is_expired():
                        self.redis_client.delete(key)
                        self.stats["expirations"] += 1
                        return None
                    
                    entry.access()
                    self.redis_client.set(
                        key, 
                        pickle.dumps(entry.to_dict()), 
                        ex=entry.ttl
                    )
                    
                    self.stats["hits"] += 1
                    return entry.value
                
                self.stats["misses"] += 1
                return None
            except Exception as e:
                logger.warning(f"Redis error in get: {e}")
        
        # Fall back to in-memory cache
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                self.stats["expirations"] += 1
                return None
            
            entry.access()
            self.stats["hits"] += 1
            return entry.value
        
        self.stats["misses"] += 1
        return None
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (defaults to cache default)
            metadata: Optional metadata
        """
        entry_ttl = ttl if ttl is not None else self.ttl
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=entry_ttl,
            metadata=metadata
        )
        
        # Try Redis first if available
        if self.redis_client:
            try:
                self.redis_client.set(
                    key, 
                    pickle.dumps(entry.to_dict()), 
                    ex=entry_ttl
                )
                return
            except Exception as e:
                logger.warning(f"Redis error in put: {e}")
        
        # Fall back to in-memory cache
        # Check if we need to evict an entry
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_entry()
        
        self.cache[key] = entry
    
    def _evict_entry(self) -> None:
        """Evict an entry based on the cache policy."""
        if not self.cache:
            return
        
        if self.policy == "LRU":
            # Least Recently Used
            key_to_evict = min(self.cache.items(), key=lambda x: x[1].last_access)[0]
        elif self.policy == "LFU":
            # Least Frequently Used
            key_to_evict = min(self.cache.items(), key=lambda x: x[1].access_count)[0]
        elif self.policy == "TLFU":
            # Time-aware Least Frequently Used
            current_time = time.time()
            key_to_evict = min(
                self.cache.items(),
                key=lambda x: (x[1].access_count / max(1, (current_time - x[1].timestamp) / 3600))
            )[0]
        else:
            # Default to random eviction
            key_to_evict = list(self.cache.keys())[0]
        
        del self.cache[key_to_evict]
        self.stats["evictions"] += 1
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was found and invalidated, False otherwise
        """
        # Try Redis first if available
        if self.redis_client:
            try:
                result = self.redis_client.delete(key)
                if result:
                    return True
            except Exception as e:
                logger.warning(f"Redis error in invalidate: {e}")
        
        # Fall back to in-memory cache
        if key in self.cache:
            del self.cache[key]
            return True
        
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Try Redis first if available
        if self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    count = self.redis_client.delete(*keys)
                    return count
            except Exception as e:
                logger.warning(f"Redis error in invalidate_pattern: {e}")
        
        # Fall back to in-memory cache
        # Simple pattern matching for in-memory cache
        keys_to_delete = []
        for key in self.cache.keys():
            if pattern in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.cache[key]
            count += 1
        
        return count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Try Redis first if available
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis error in clear: {e}")
        
        # Fall back to in-memory cache
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = self.stats.copy()
        stats["size"] = len(self.cache)
        stats["hit_ratio"] = stats["hits"] / max(1, stats["hits"] + stats["misses"])
        return stats


class CachedRetriever(Retriever):
    """
    Cached retriever that wraps another retriever.
    """
    def __init__(
        self,
        base_retriever: Retriever,
        cache_system: Optional[CacheSystem] = None
    ):
        """
        Initialize the cached retriever.
        
        Args:
            base_retriever: Base retriever to wrap
            cache_system: Cache system to use
        """
        self.base_retriever = base_retriever
        self.cache_system = cache_system or CacheSystem()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.
        
        Args:
            documents: List of documents to add
        """
        # Add to base retriever
        self.base_retriever.add_documents(documents)
        
        # Invalidate cache entries that might be affected
        # This is a simplified approach - a more sophisticated implementation
        # would be more selective about which cache entries to invalidate
        self._invalidate_affected_cache_entries(documents)
    
    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query to retrieve documents for
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieval result containing documents and scores
        """
        # Generate cache key
        cache_key = CacheKey.generate_key(query, {"top_k": top_k})
        
        # Check cache
        cached_result = self.cache_system.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for query: {query.text}")
            return cached_result
        
        # Cache miss - use base retriever
        logger.info(f"Cache miss for query: {query.text}")
        result = self.base_retriever.retrieve(query, top_k=top_k)
        
        # Cache the result
        self.cache_system.put(
            key=cache_key,
            value=result,
            metadata={"query": query.text, "top_k": top_k}
        )
        
        return result
    
    def update_documents(self, documents: List[Document]) -> None:
        """
        Update existing documents in the retriever.
        
        Args:
            documents: List of documents to update
        """
        # Update in base retriever
        self.base_retriever.update_documents(documents)
        
        # Invalidate cache entries that might be affected
        self._invalidate_affected_cache_entries(documents)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the retriever.
        
        Args:
            document_ids: List of document IDs to delete
        """
        # Delete from base retriever
        self.base_retriever.delete_documents(document_ids)
        
        # Invalidate cache entries that might be affected
        self._invalidate_affected_cache_entries_by_ids(document_ids)
    
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
    
    def _invalidate_affected_cache_entries_by_ids(self, document_ids: List[str]) -> None:
        """
        Invalidate cache entries that might be affected by document deletions.
        
        Args:
            document_ids: List of document IDs that were deleted
        """
        # This is a simplified approach - a more sophisticated implementation
        # would be more selective about which cache entries to invalidate
        
        # For simplicity, just clear the entire cache
        # In a real implementation, we would track which documents are in which cache entries
        self.cache_system.clear()


class CachedGenerator(Generator):
    """
    Cached generator that wraps another generator.
    """
    def __init__(
        self,
        base_generator: Generator,
        cache_system: Optional[CacheSystem] = None
    ):
        """
        Initialize the cached generator.
        
        Args:
            base_generator: Base generator to wrap
            cache_system: Cache system to use
        """
        self.base_generator = base_generator
        self.cache_system = cache_system or CacheSystem()
    
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
        # Generate cache key
        # We include document IDs and scores in the key to ensure cache hits
        # only occur for the same set of retrieved documents
        doc_ids_scores = [
            (doc.id, score) 
            for doc, score in zip(retrieval_result.documents, retrieval_result.scores)
        ]
        cache_context = {
            "doc_ids_scores": doc_ids_scores
        }
        cache_key = CacheKey.generate_key(query, cache_context)
        
        # Check cache
        cached_result = self.cache_system.get(cache_key)
        if cached_result is not None:
            logger.info(f"Generator cache hit for query: {query.text}")
            return cached_result
        
        # Cache miss - use base generator
        logger.info(f"Generator cache miss for query: {query.text}")
        result = self.base_generator.generate(query, retrieval_result)
        
        # Cache the result
        self.cache_system.put(
            key=cache_key,
            value=result,
            metadata={"query": query.text}
        )
        
        return result


class CacheRAG(RAGSystem):
    """
    Cache RAG system for network security packet analysis.
    """
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        base_retriever: Optional[Retriever] = None,
        base_generator: Optional[Generator] = None,
        retriever_cache: Optional[CacheSystem] = None,
        generator_cache: Optional[CacheSystem] = None
    ):
        """
        Initialize the Cache RAG system.
        
        Args:
            embedding_model: Model for encoding documents and queries
            base_retriever: Base retriever to use
            base_generator: Base generator to use
            retriever_cache: Cache system for retriever
            generator_cache: Cache system for generator
        """
        self.embedding_model = embedding_model or NetworkEmbeddingModel()
        
        # Set up base components if not provided
        if base_retriever is None:
            base_retriever = VectorRetriever(embedding_model=self.embedding_model)
        
        if base_generator is None:
            base_generator = NetworkSecurityGenerator()
        
        # Set up cache systems
        self.retriever_cache = retriever_cache or CacheSystem(
            max_size=CACHE_CONFIG["levels"]["L2"]["size"],
            ttl=CACHE_CONFIG["levels"]["L2"]["ttl"],
            policy=CACHE_CONFIG["policy"]
        )
        
        self.generator_cache = generator_cache or CacheSystem(
            max_size=CACHE_CONFIG["levels"]["L1"]["size"],
            ttl=CACHE_CONFIG["levels"]["L1"]["ttl"],
            policy=CACHE_CONFIG["policy"]
        )
        
        # Set up cached components
        self.retriever = CachedRetriever(
            base_retriever=base_retriever,
            cache_system=self.retriever_cache
        )
        
        self.generator = CachedGenerator(
            base_generator=base_generator,
            cache_system=self.generator_cache
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
        cached_result = self.generator_cache.get(cache_key)
        
        if cached_result is not None:
            logger.info(f"Final result cache hit for query: {query.text}")
            return cached_result
        
        # No final result cache hit, proceed with retrieval and generation
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(query, top_k=top_k)
        
        # Generate response
        response = self.generator.generate(query, retrieval_result)
        
        # Add cache performance information
        retriever_stats = self.retriever_cache.get_stats()
        generator_stats = self.generator_cache.get_stats()
        
        response["cache_info"] = {
            "retriever_cache": {
                "hit_ratio": retriever_stats["hit_ratio"],
                "size": retriever_stats["size"]
            },
            "generator_cache": {
                "hit_ratio": generator_stats["hit_ratio"],
                "size": generator_stats["size"]
            },
            "processing_time": time.time() - start_time
        }
        
        # Cache the final result
        self.generator_cache.put(
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
        
        # Calculate metrics
        # This is a placeholder - actual metrics would depend on the specific evaluation task
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "latency": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "cache_hit_ratio": self.retriever_cache.get_stats()["hit_ratio"]
        }
        
        return metrics
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.retriever_cache.clear()
        self.generator_cache.clear()
        logger.info("All caches cleared")
