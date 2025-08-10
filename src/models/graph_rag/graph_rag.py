"""
Implementation of the Graph RAG system for network security packet analysis.
"""

import os
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import time

from ..common.base import Document, Query, RetrievalResult, EmbeddingModel, Retriever, Generator, RAGSystem
from ..common.config import GRAPH_DB_CONFIG
from ..traditional_rag.traditional_rag import NetworkEmbeddingModel, NetworkSecurityGenerator
from ..utils.network_utils import format_packet_data, generate_flow_id, extract_protocol_features

# Configure logging
logger = logging.getLogger(__name__)

class NetworkNode:
    """
    Class representing a node in the network graph.
    """
    def __init__(
        self,
        id: str,
        type: str,
        properties: Dict[str, Any]
    ):
        """
        Initialize a network node.
        
        Args:
            id: Unique identifier for the node
            type: Type of the node (ip_address, port, protocol, etc.)
            properties: Node properties
        """
        self.id = id
        self.type = type
        self.properties = properties
        self.embedding: Optional[np.ndarray] = None
    
    def __str__(self) -> str:
        return f"NetworkNode(id={self.id}, type={self.type})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkNode':
        """Create node from dictionary representation."""
        return cls(
            id=data["id"],
            type=data["type"],
            properties=data["properties"]
        )


class NetworkEdge:
    """
    Class representing an edge in the network graph.
    """
    def __init__(
        self,
        source_id: str,
        target_id: str,
        type: str,
        properties: Dict[str, Any]
    ):
        """
        Initialize a network edge.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            type: Type of the edge (connects_to, uses, etc.)
            properties: Edge properties
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.properties = properties
    
    def __str__(self) -> str:
        return f"NetworkEdge(source={self.source_id}, target={self.target_id}, type={self.type})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkEdge':
        """Create edge from dictionary representation."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            properties=data["properties"]
        )


class NetworkGraph:
    """
    Class representing the network graph.
    Uses NetworkX for graph operations.
    """
    def __init__(self):
        """Initialize the network graph."""
        self.graph = nx.MultiDiGraph()
        self.node_types = set(GRAPH_DB_CONFIG["node_types"])
        self.edge_types = set(GRAPH_DB_CONFIG["edge_types"])
        self.default_weight = GRAPH_DB_CONFIG["default_weight"]
        self.temporal_window = GRAPH_DB_CONFIG["temporal_window"]
        
        # Node and edge lookup
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        
        # Document mapping
        self.document_to_subgraph: Dict[str, Set[str]] = {}
    
    def add_node(self, node: NetworkNode) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
        """
        if node.type not in self.node_types:
            logger.warning(f"Unknown node type: {node.type}")
        
        # Add to NetworkX graph
        self.graph.add_node(
            node.id,
            type=node.type,
            **node.properties
        )
        
        # Store in lookup
        self.nodes[node.id] = node
    
    def add_edge(self, edge: NetworkEdge) -> None:
        """
        Add an edge to the graph.
        
        Args:
            edge: Edge to add
        """
        if edge.type not in self.edge_types:
            logger.warning(f"Unknown edge type: {edge.type}")
        
        # Add to NetworkX graph
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            type=edge.type,
            weight=edge.properties.get("weight", self.default_weight),
            **edge.properties
        )
        
        # Store in lookup
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[NetworkNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: ID of the node to get
            
        Returns:
            Node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None) -> List[str]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: ID of the node
            edge_types: Optional list of edge types to filter by
            
        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []
        
        neighbors = []
        
        for neighbor in self.graph.neighbors(node_id):
            edges = self.graph.get_edge_data(node_id, neighbor)
            
            for key, data in edges.items():
                if edge_types is None or data["type"] in edge_types:
                    neighbors.append(neighbor)
                    break
        
        return neighbors
    
    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> nx.MultiDiGraph:
        """
        Get a subgraph centered on the specified nodes.
        
        Args:
            node_ids: List of node IDs to center the subgraph on
            depth: Maximum depth of the subgraph
            
        Returns:
            NetworkX subgraph
        """
        # Start with the specified nodes
        nodes_to_include = set(node_ids)
        
        # Expand to neighbors up to the specified depth
        current_frontier = set(node_ids)
        
        for _ in range(depth):
            next_frontier = set()
            
            for node_id in current_frontier:
                neighbors = self.get_neighbors(node_id)
                next_frontier.update(neighbors)
            
            nodes_to_include.update(next_frontier)
            current_frontier = next_frontier
        
        # Extract the subgraph
        return self.graph.subgraph(nodes_to_include)
    
    def find_paths(
        self, 
        source_id: str, 
        target_id: str, 
        max_length: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                self.graph, 
                source=source_id, 
                target=target_id, 
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXError:
            return []
    
    def find_nodes_by_property(
        self, 
        property_name: str, 
        property_value: Any, 
        node_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Find nodes by property value.
        
        Args:
            property_name: Name of the property to match
            property_value: Value of the property to match
            node_types: Optional list of node types to filter by
            
        Returns:
            List of matching node IDs
        """
        matching_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if property_name in node_data and node_data[property_name] == property_value:
                if node_types is None or node_data["type"] in node_types:
                    matching_nodes.append(node_id)
        
        return matching_nodes
    
    def map_document_to_subgraph(self, document_id: str, node_ids: List[str]) -> None:
        """
        Map a document to a subgraph.
        
        Args:
            document_id: ID of the document
            node_ids: List of node IDs in the subgraph
        """
        self.document_to_subgraph[document_id] = set(node_ids)
    
    def get_document_subgraph(self, document_id: str) -> Set[str]:
        """
        Get the subgraph associated with a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Set of node IDs in the subgraph
        """
        return self.document_to_subgraph.get(document_id, set())
    
    def get_documents_for_node(self, node_id: str) -> List[str]:
        """
        Get documents associated with a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of document IDs
        """
        document_ids = []
        
        for doc_id, node_ids in self.document_to_subgraph.items():
            if node_id in node_ids:
                document_ids.append(doc_id)
        
        return document_ids
    
    def compute_node_centrality(self, node_id: str) -> float:
        """
        Compute the centrality of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Centrality score
        """
        if node_id not in self.graph:
            return 0.0
        
        # Use degree centrality as a simple measure
        centrality = nx.degree_centrality(self.graph)
        return centrality.get(node_id, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "document_to_subgraph": {
                doc_id: list(node_ids) 
                for doc_id, node_ids in self.document_to_subgraph.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkGraph':
        """Create graph from dictionary representation."""
        graph = cls()
        
        # Add nodes
        for node_data in data["nodes"]:
            node = NetworkNode.from_dict(node_data)
            graph.add_node(node)
        
        # Add edges
        for edge_data in data["edges"]:
            edge = NetworkEdge.from_dict(edge_data)
            graph.add_edge(edge)
        
        # Add document mappings
        for doc_id, node_ids in data["document_to_subgraph"].items():
            graph.document_to_subgraph[doc_id] = set(node_ids)
        
        return graph


class GraphBuilder:
    """
    Class for building a network graph from documents.
    """
    def __init__(self, graph: Optional[NetworkGraph] = None):
        """
        Initialize the graph builder.
        
        Args:
            graph: Optional existing graph to build upon
        """
        self.graph = graph or NetworkGraph()
    
    def build_from_documents(self, documents: List[Document]) -> NetworkGraph:
        """
        Build a graph from documents.
        
        Args:
            documents: List of documents to build the graph from
            
        Returns:
            The built graph
        """
        for document in documents:
            self._process_document(document)
        
        return self.graph
    
    def _process_document(self, document: Document) -> None:
        """
        Process a document and add its entities to the graph.
        
        Args:
            document: Document to process
        """
        metadata = document.metadata
        document_nodes = set()
        
        # Process IP addresses
        src_ip = metadata.get("src_ip")
        dst_ip = metadata.get("dst_ip")
        
        if src_ip:
            src_ip_node_id = f"ip:{src_ip}"
            if src_ip_node_id not in self.graph.nodes:
                src_ip_node = NetworkNode(
                    id=src_ip_node_id,
                    type="ip_address",
                    properties={
                        "address": src_ip,
                        "first_seen": metadata.get("timestamp", time.time()),
                        "is_source": True
                    }
                )
                self.graph.add_node(src_ip_node)
            document_nodes.add(src_ip_node_id)
        
        if dst_ip:
            dst_ip_node_id = f"ip:{dst_ip}"
            if dst_ip_node_id not in self.graph.nodes:
                dst_ip_node = NetworkNode(
                    id=dst_ip_node_id,
                    type="ip_address",
                    properties={
                        "address": dst_ip,
                        "first_seen": metadata.get("timestamp", time.time()),
                        "is_destination": True
                    }
                )
                self.graph.add_node(dst_ip_node)
            document_nodes.add(dst_ip_node_id)
        
        # Process ports
        src_port = metadata.get("src_port")
        dst_port = metadata.get("dst_port")
        
        if src_port:
            src_port_node_id = f"port:{src_port}"
            if src_port_node_id not in self.graph.nodes:
                src_port_node = NetworkNode(
                    id=src_port_node_id,
                    type="port",
                    properties={
                        "number": src_port,
                        "is_source": True
                    }
                )
                self.graph.add_node(src_port_node)
            document_nodes.add(src_port_node_id)
        
        if dst_port:
            dst_port_node_id = f"port:{dst_port}"
            if dst_port_node_id not in self.graph.nodes:
                dst_port_node = NetworkNode(
                    id=dst_port_node_id,
                    type="port",
                    properties={
                        "number": dst_port,
                        "is_destination": True
                    }
                )
                self.graph.add_node(dst_port_node)
            document_nodes.add(dst_port_node_id)
        
        # Process protocol
        protocol = metadata.get("protocol")
        if protocol:
            protocol_node_id = f"protocol:{protocol}"
            if protocol_node_id not in self.graph.nodes:
                protocol_node = NetworkNode(
                    id=protocol_node_id,
                    type="protocol",
                    properties={
                        "name": protocol
                    }
                )
                self.graph.add_node(protocol_node)
            document_nodes.add(protocol_node_id)
        
        # Create session node
        session_node_id = f"session:{document.id}"
        session_node = NetworkNode(
            id=session_node_id,
            type="session",
            properties={
                "document_id": document.id,
                "timestamp": metadata.get("timestamp", time.time()),
                "duration": metadata.get("duration", 0)
            }
        )
        self.graph.add_node(session_node)
        document_nodes.add(session_node_id)
        
        # Create edges
        
        # IP to IP connection
        if src_ip and dst_ip:
            connection_edge = NetworkEdge(
                source_id=f"ip:{src_ip}",
                target_id=f"ip:{dst_ip}",
                type="connects_to",
                properties={
                    "timestamp": metadata.get("timestamp", time.time()),
                    "document_id": document.id,
                    "weight": 1.0
                }
            )
            self.graph.add_edge(connection_edge)
        
        # IP to port edges
        if src_ip and src_port:
            src_uses_edge = NetworkEdge(
                source_id=f"ip:{src_ip}",
                target_id=f"port:{src_port}",
                type="uses",
                properties={
                    "timestamp": metadata.get("timestamp", time.time()),
                    "document_id": document.id,
                    "weight": 1.0
                }
            )
            self.graph.add_edge(src_uses_edge)
        
        if dst_ip and dst_port:
            dst_uses_edge = NetworkEdge(
                source_id=f"ip:{dst_ip}",
                target_id=f"port:{dst_port}",
                type="uses",
                properties={
                    "timestamp": metadata.get("timestamp", time.time()),
                    "document_id": document.id,
                    "weight": 1.0
                }
            )
            self.graph.add_edge(dst_uses_edge)
        
        # Protocol edges
        if protocol:
            if src_ip:
                src_protocol_edge = NetworkEdge(
                    source_id=f"ip:{src_ip}",
                    target_id=f"protocol:{protocol}",
                    type="uses",
                    properties={
                        "timestamp": metadata.get("timestamp", time.time()),
                        "document_id": document.id,
                        "weight": 1.0
                    }
                )
                self.graph.add_edge(src_protocol_edge)
            
            if dst_ip:
                dst_protocol_edge = NetworkEdge(
                    source_id=f"ip:{dst_ip}",
                    target_id=f"protocol:{protocol}",
                    type="uses",
                    properties={
                        "timestamp": metadata.get("timestamp", time.time()),
                        "document_id": document.id,
                        "weight": 1.0
                    }
                )
                self.graph.add_edge(dst_protocol_edge)
        
        # Session edges
        if src_ip:
            session_src_edge = NetworkEdge(
                source_id=session_node_id,
                target_id=f"ip:{src_ip}",
                type="contains",
                properties={
                    "timestamp": metadata.get("timestamp", time.time()),
                    "document_id": document.id,
                    "weight": 1.0
                }
            )
            self.graph.add_edge(session_src_edge)
        
        if dst_ip:
            session_dst_edge = NetworkEdge(
                source_id=session_node_id,
                target_id=f"ip:{dst_ip}",
                type="contains",
                properties={
                    "timestamp": metadata.get("timestamp", time.time()),
                    "document_id": document.id,
                    "weight": 1.0
                }
            )
            self.graph.add_edge(session_dst_edge)
        
        # Map document to subgraph
        self.graph.map_document_to_subgraph(document.id, document_nodes)


class GraphRetriever(Retriever):
    """
    Graph-based retriever for network security data.
    """
    def __init__(
        self,
        graph: Optional[NetworkGraph] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the graph retriever.
        
        Args:
            graph: Network graph
            embedding_model: Model for encoding queries
        """
        self.graph = graph or NetworkGraph()
        self.embedding_model = embedding_model or NetworkEmbeddingModel()
        self.graph_builder = GraphBuilder(self.graph)
        
        # Document storage
        self.documents: Dict[str, Document] = {}
    
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
        
        logger.info(f"Added {len(documents)} documents to graph retriever. Total: {len(self.documents)}")
    
    def retrieve(self, query: Query, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.
        
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
                "nodes": relevant_nodes
            }
        )
    
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
        
        logger.info(f"Deleted {len(document_ids)} documents from graph retriever. Remaining: {len(self.documents)}")


class GraphRAG(RAGSystem):
    """
    Graph RAG system for network security packet analysis.
    """
    def __init__(
        self,
        graph: Optional[NetworkGraph] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None
    ):
        """
        Initialize the Graph RAG system.
        
        Args:
            graph: Network graph
            embedding_model: Model for encoding documents and queries
            retriever: Retriever component
            generator: Generator component
        """
        self.embedding_model = embedding_model or NetworkEmbeddingModel()
        self.graph = graph or NetworkGraph()
        self.retriever = retriever or GraphRetriever(
            graph=self.graph,
            embedding_model=self.embedding_model
        )
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
        
        # Enhance response with graph information
        if "entities" in retrieval_result.metadata and "nodes" in retrieval_result.metadata:
            response["graph_context"] = {
                "entities": retrieval_result.metadata["entities"],
                "nodes": retrieval_result.metadata["nodes"]
            }
        
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
