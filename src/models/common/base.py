"""
Base classes and utilities for RAG systems in network security packet analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Document:
    """
    Base class representing a document in the RAG system.
    For network security, a document typically represents a network flow or packet.
    """
    def __init__(
        self, 
        id: str,
        content: str,
        metadata: Dict[str, Any]
    ):
        """
        Initialize a document.
        
        Args:
            id: Unique identifier for the document
            content: Text representation of the network data
            metadata: Additional information about the network data
        """
        self.id = id
        self.content = content
        self.metadata = metadata
        self.embedding: Optional[np.ndarray] = None
    
    def __str__(self) -> str:
        return f"Document(id={self.id}, metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary representation."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data["metadata"]
        )


class Query:
    """
    Class representing a query in the RAG system.
    For network security, a query typically represents a security question or alert.
    """
    def __init__(
        self,
        text: str,
        filters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a query.
        
        Args:
            text: The query text
            filters: Optional filters to apply during retrieval
            metadata: Additional information about the query
        """
        self.text = text
        self.filters = filters or {}
        self.metadata = metadata or {}
        self.embedding: Optional[np.ndarray] = None
    
    def __str__(self) -> str:
        return f"Query(text={self.text}, filters={self.filters})"


class RetrievalResult:
    """
    Class representing the result of a retrieval operation.
    """
    def __init__(
        self,
        documents: List[Document],
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a retrieval result.
        
        Args:
            documents: List of retrieved documents
            scores: Relevance scores for each document
            metadata: Additional information about the retrieval
        """
        self.documents = documents
        self.scores = scores
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def get_top_k(self, k: int) -> 'RetrievalResult':
        """Get top k documents from the result."""
        if k >= len(self.documents):
            return self
        
        return RetrievalResult(
            documents=self.documents[:k],
            scores=self.scores[:k],
            metadata=self.metadata
        )


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    """
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        pass
    
    @abstractmethod
    def encode_documents(self, documents: List[Document]) -> List[Document]:
        """
        Encode documents and update their embedding attribute.
        
        Args:
            documents: List of documents to encode
            
        Returns:
            List of documents with embeddings
        """
        pass
    
    @abstractmethod
    def encode_query(self, query: Query) -> Query:
        """
        Encode query and update its embedding attribute.
        
        Args:
            query: Query to encode
            
        Returns:
            Query with embedding
        """
        pass


class Retriever(ABC):
    """
    Abstract base class for retrievers.
    """
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.
        
        Args:
            documents: List of documents to add
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query to retrieve documents for
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieval result containing documents and scores
        """
        pass
    
    @abstractmethod
    def update_documents(self, documents: List[Document]) -> None:
        """
        Update existing documents in the retriever.
        
        Args:
            documents: List of documents to update
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the retriever.
        
        Args:
            document_ids: List of document IDs to delete
        """
        pass


class Generator(ABC):
    """
    Abstract base class for generators.
    """
    @abstractmethod
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
        pass


class RAGSystem(ABC):
    """
    Abstract base class for RAG systems.
    """
    @abstractmethod
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
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Union[Dict[str, Any], Document]]) -> None:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of documents to add
        """
        pass
    
    @abstractmethod
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
        pass
