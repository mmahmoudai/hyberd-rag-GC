"""
Test scenario generator for RAG systems in network security packet analysis.
"""

import os
import random
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

from ..common.base import Document, Query
from ..common.config import TEST_SCENARIO_CONFIG
from .dataset_processing import CICIDSProcessor, UNSWProcessor, CustomPCAPProcessor

# Configure logging
logger = logging.getLogger(__name__)

class TestScenarioGenerator:
    """
    Base class for test scenario generators.
    """
    def __init__(
        self,
        scenario_name: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the test scenario generator.
        
        Args:
            scenario_name: Name of the test scenario
            output_dir: Directory to save generated scenarios
        """
        self.scenario_name = scenario_name
        self.output_dir = output_dir or Path(f"/home/ubuntu/research/experiments/scenarios/{scenario_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate the test scenario.
        
        Returns:
            Dictionary containing the generated scenario
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, scenario: Dict[str, Any]) -> None:
        """
        Save the generated scenario to disk.
        
        Args:
            scenario: Generated scenario
        """
        output_file = os.path.join(self.output_dir, f"{self.scenario_name}.json")
        
        with open(output_file, "w") as f:
            json.dump(scenario, f, indent=2)
        
        logger.info(f"Saved scenario to {output_file}")
    
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load a previously generated scenario from disk.
        
        Returns:
            Loaded scenario or None if not found
        """
        input_file = os.path.join(self.output_dir, f"{self.scenario_name}.json")
        
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                scenario = json.load(f)
            
            logger.info(f"Loaded scenario from {input_file}")
            return scenario
        
        logger.warning(f"Scenario file not found: {input_file}")
        return None


class StandardScenarioGenerator(TestScenarioGenerator):
    """
    Generator for standard test scenarios.
    """
    def __init__(
        self,
        dataset_name: str,
        num_queries: int = 100,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the standard scenario generator.
        
        Args:
            dataset_name: Name of the dataset to use
            num_queries: Number of queries to generate
            output_dir: Directory to save generated scenarios
        """
        super().__init__(f"standard_{dataset_name}", output_dir)
        self.dataset_name = dataset_name
        self.num_queries = num_queries
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate a standard test scenario.
        
        Returns:
            Dictionary containing the generated scenario
        """
        # Check if scenario already exists
        existing_scenario = self.load()
        if existing_scenario:
            return existing_scenario
        
        # Load dataset
        if self.dataset_name == "cic_ids2017":
            processor = CICIDSProcessor()
        elif self.dataset_name == "unsw_nb15":
            processor = UNSWProcessor()
        elif self.dataset_name == "custom_pcap":
            processor = CustomPCAPProcessor()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Process dataset
        splits = processor.process()
        
        # Use test split for evaluation
        test_docs = splits.get("test", [])
        
        if not test_docs:
            logger.warning(f"No test documents found for {self.dataset_name}")
            test_docs = splits.get("val", [])
            
            if not test_docs:
                logger.warning(f"No validation documents found for {self.dataset_name}")
                test_docs = splits.get("train", [])
                
                if not test_docs:
                    logger.error(f"No documents found for {self.dataset_name}")
                    raise ValueError(f"No documents found for {self.dataset_name}")
        
        logger.info(f"Generating standard scenario with {len(test_docs)} test documents")
        
        # Generate queries
        queries = self._generate_queries(test_docs)
        
        # Generate ground truth
        ground_truth = self._generate_ground_truth(queries, test_docs)
        
        # Create scenario
        scenario = {
            "name": self.scenario_name,
            "dataset": self.dataset_name,
            "num_queries": len(queries),
            "num_documents": len(test_docs),
            "queries": queries,
            "ground_truth": ground_truth
        }
        
        # Save scenario
        self.save(scenario)
        
        return scenario
    
    def _generate_queries(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate queries from documents.
        
        Args:
            documents: List of documents to generate queries from
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Group documents by attack type
        attack_groups = {}
        benign_docs = []
        
        for doc in documents:
            if doc.metadata.get("is_attack", False):
                attack_type = doc.metadata.get("attack_type", "unknown")
                if attack_type not in attack_groups:
                    attack_groups[attack_type] = []
                attack_groups[attack_type].append(doc)
            else:
                benign_docs.append(doc)
        
        # Generate queries for each attack type
        for attack_type, attack_docs in attack_groups.items():
            # Determine number of queries for this attack type
            num_queries = max(1, int(self.num_queries * (len(attack_docs) / len(documents))))
            
            # Generate queries
            for i in range(num_queries):
                # Select a random document
                doc = random.choice(attack_docs)
                
                # Generate query text
                query_text = self._generate_query_text(doc, is_attack=True)
                
                # Create query
                query = {
                    "id": f"{attack_type}_{i}",
                    "text": query_text,
                    "type": "attack",
                    "attack_type": attack_type,
                    "document_id": doc.id
                }
                
                queries.append(query)
        
        # Generate queries for benign traffic
        num_benign_queries = self.num_queries - len(queries)
        
        for i in range(num_benign_queries):
            if benign_docs:
                # Select a random document
                doc = random.choice(benign_docs)
                
                # Generate query text
                query_text = self._generate_query_text(doc, is_attack=False)
                
                # Create query
                query = {
                    "id": f"benign_{i}",
                    "text": query_text,
                    "type": "benign",
                    "document_id": doc.id
                }
                
                queries.append(query)
        
        # Shuffle queries
        random.shuffle(queries)
        
        logger.info(f"Generated {len(queries)} queries")
        return queries
    
    def _generate_query_text(self, document: Document, is_attack: bool) -> str:
        """
        Generate query text from a document.
        
        Args:
            document: Document to generate query from
            is_attack: Whether the document represents an attack
            
        Returns:
            Generated query text
        """
        # Extract metadata
        src_ip = document.metadata.get("src_ip", "")
        dst_ip = document.metadata.get("dst_ip", "")
        src_port = document.metadata.get("src_port", "")
        dst_port = document.metadata.get("dst_port", "")
        protocol = document.metadata.get("protocol", "")
        
        # Generate query templates
        templates = []
        
        if is_attack:
            attack_type = document.metadata.get("attack_type", "")
            
            templates = [
                f"Analyze traffic from {src_ip} to {dst_ip}",
                f"Check for suspicious activity involving {src_ip}",
                f"Investigate {protocol} traffic to port {dst_port}",
                f"Examine network flow between {src_ip} and {dst_ip}",
                f"Analyze packets with source port {src_port}",
                f"Look for potential threats in traffic to {dst_ip}",
                f"Investigate unusual {protocol} patterns",
                f"Check for attack signatures in traffic from {src_ip}",
                f"Analyze security events involving {dst_ip}:{dst_port}",
                f"Examine traffic patterns for {src_ip} to {dst_ip}"
            ]
            
            # Add attack-specific templates
            if "scan" in attack_type.lower() or "reconnaissance" in attack_type.lower():
                templates.extend([
                    f"Check for port scanning activity from {src_ip}",
                    f"Investigate potential network scanning to {dst_ip}",
                    f"Analyze reconnaissance attempts on port {dst_port}",
                    f"Look for scanning patterns in {protocol} traffic"
                ])
            elif "dos" in attack_type.lower() or "ddos" in attack_type.lower():
                templates.extend([
                    f"Check for DoS attack from {src_ip}",
                    f"Investigate potential DDoS on {dst_ip}:{dst_port}",
                    f"Analyze high-volume traffic to {dst_ip}",
                    f"Look for service disruption patterns on {dst_port}"
                ])
            elif "brute" in attack_type.lower():
                templates.extend([
                    f"Check for brute force attempts on {dst_ip}",
                    f"Investigate login attempts to {dst_ip}:{dst_port}",
                    f"Analyze authentication failures on {dst_ip}",
                    f"Look for credential stuffing patterns to {dst_port}"
                ])
        else:
            templates = [
                f"Show traffic between {src_ip} and {dst_ip}",
                f"Display {protocol} connections to {dst_ip}",
                f"List network flows involving {src_ip}",
                f"Show packets to port {dst_port}",
                f"Display traffic summary for {src_ip}",
                f"List recent connections to {dst_ip}",
                f"Show {protocol} traffic statistics",
                f"Summarize network activity for {src_ip}",
                f"Display connection details for {dst_ip}:{dst_port}",
                f"Show traffic patterns between {src_ip} and {dst_ip}"
            ]
        
        # Select a random template
        query_text = random.choice(templates)
        
        return query_text
    
    def _generate_ground_truth(
        self,
        queries: List[Dict[str, Any]],
        documents: List[Document]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate ground truth for queries.
        
        Args:
            queries: List of queries
            documents: List of documents
            
        Returns:
            Dictionary mapping query IDs to lists of relevant document IDs
        """
        ground_truth = {}
        
        # Create document lookup
        doc_lookup = {doc.id: doc for doc in documents}
        
        # Process each query
        for query in queries:
            query_id = query["id"]
            document_id = query.get("document_id")
            
            # Start with the document that generated the query
            relevant_docs = []
            
            if document_id and document_id in doc_lookup:
                relevant_docs.append(doc_lookup[document_id])
            
            # Find additional relevant documents
            if "src_ip" in query["text"] or "dst_ip" in query["text"]:
                # Extract IPs from query text
                query_text = query["text"]
                ips = []
                
                for word in query_text.split():
                    if word.count('.') == 3 and all(part.isdigit() for part in word.split('.')):
                        ips.append(word)
                
                # Find documents with matching IPs
                for doc in documents:
                    if doc.id != document_id:  # Skip the original document
                        src_ip = doc.metadata.get("src_ip", "")
                        dst_ip = doc.metadata.get("dst_ip", "")
                        
                        if any(ip in [src_ip, dst_ip] for ip in ips):
                            relevant_docs.append(doc)
            
            # Limit number of relevant documents
            if len(relevant_docs) > 20:
                # Keep the original document and sample from the rest
                original_doc = relevant_docs[0]
                other_docs = relevant_docs[1:]
                sampled_docs = random.sample(other_docs, 19)
                relevant_docs = [original_doc] + sampled_docs
            
            # Convert to dictionaries
            ground_truth[query_id] = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]
        
        logger.info(f"Generated ground truth for {len(queries)} queries")
        return ground_truth


class ZeroDayScenarioGenerator(TestScenarioGenerator):
    """
    Generator for zero-day attack test scenarios.
    """
    def __init__(
        self,
        num_queries: int = 50,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the zero-day scenario generator.
        
        Args:
            num_queries: Number of queries to generate
            output_dir: Directory to save generated scenarios
        """
        super().__init__("zero_day", output_dir)
        self.num_queries = num_queries
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate a zero-day attack test scenario.
        
        Returns:
            Dictionary containing the generated scenario
        """
        # Check if scenario already exists
        existing_scenario = self.load()
        if existing_scenario:
            return existing_scenario
        
        # Load datasets
        cic_processor = CICIDSProcessor()
        unsw_processor = UNSWProcessor()
        custom_processor = CustomPCAPProcessor()
        
        # Process datasets
        cic_splits = cic_processor.process()
        unsw_splits = unsw_processor.process()
        custom_splits = custom_processor.process()
        
        # Use test splits for evaluation
        cic_test_docs = cic_splits.get("test", [])
        unsw_test_docs = unsw_splits.get("test", [])
        custom_test_docs = custom_splits.get("test", [])
        
        # Combine documents
        all_docs = cic_test_docs + unsw_test_docs + custom_test_docs
        
        if not all_docs:
            logger.error("No documents found for zero-day scenario")
            raise ValueError("No documents found for zero-day scenario")
        
        logger.info(f"Generating zero-day scenario with {len(all_docs)} documents")
        
        # Create synthetic zero-day attacks
        zero_day_docs = self._create_zero_day_attacks(all_docs)
        
        # Generate queries
        queries = self._generate_queries(zero_day_docs)
        
        # Generate ground truth
        ground_truth = self._generate_ground_truth(queries, zero_day_docs)
        
        # Create scenario
        scenario = {
            "name": self.scenario_name,
            "num_queries": len(queries),
            "num_documents": len(zero_day_docs),
            "queries": queries,
            "ground_truth": ground_truth
        }
        
        # Save scenario
        self.save(scenario)
        
        return scenario
    
    def _create_zero_day_attacks(self, documents: List[Document]) -> List[Document]:
        """
        Create synthetic zero-day attacks.
        
        Args:
            documents: List of documents to base attacks on
            
        Returns:
            List of documents with synthetic zero-day attacks
        """
        # Define zero-day attack types
        zero_day_types = [
            "Novel Protocol Exploitation",
            "Firmware Vulnerability",
            "Supply Chain Compromise",
            "API Manipulation",
            "Kernel Exploitation",
            "Container Escape",
            "Hypervisor Vulnerability",
            "Microarchitectural Attack",
            "Encrypted Traffic Exploitation",
            "IoT Botnet"
        ]
        
        # Create synthetic attacks
        zero_day_docs = []
        
        # Select a subset of documents to modify
        num_zero_day = min(self.num_queries * 2, len(documents) // 10)
        selected_docs = random.sample(documents, num_zero_day)
        
        for i, doc in enumerate(selected_docs):
            # Create a copy of the document
            zero_day_doc = Document(
                id=f"zero_day_{i}",
                content=doc.content,
                metadata=doc.metadata.copy()
            )
            
            # Assign a zero-day attack type
            attack_type = random.choice(zero_day_types)
            
            # Modify metadata
            zero_day_doc.metadata["is_attack"] = True
            zero_day_doc.metadata["attack_type"] = attack_type
            zero_day_doc.metadata["zero_day"] = True
            
            # Add threat indicators
            threat_indicators = zero_day_doc.metadata.get("threat_indicators", [])
            threat_indicators.append(f"Zero-day attack: {attack_type}")
            zero_day_doc.metadata["threat_indicators"] = threat_indicators
            
            # Add attack-specific metadata
            if "Protocol" in attack_type:
                zero_day_doc.metadata["protocol_anomaly"] = True
                zero_day_doc.metadata["anomaly_score"] = random.uniform(0.8, 1.0)
            elif "Firmware" in attack_type:
                zero_day_doc.metadata["firmware_target"] = random.choice(["Router", "Switch", "IoT Device"])
                zero_day_doc.metadata["exploitation_phase"] = random.choice(["Reconnaissance", "Exploitation", "Persistence"])
            elif "Supply Chain" in attack_type:
                zero_day_doc.metadata["compromised_component"] = random.choice(["Library", "Framework", "Plugin"])
                zero_day_doc.metadata["backdoor_type"] = random.choice(["Command Injection", "Data Exfiltration", "Authentication Bypass"])
            elif "API" in attack_type:
                zero_day_doc.metadata["api_target"] = random.choice(["REST", "GraphQL", "SOAP"])
                zero_day_doc.metadata["manipulation_type"] = random.choice(["Parameter Tampering", "Rate Limiting Bypass", "Authentication Bypass"])
            elif "Kernel" in attack_type:
                zero_day_doc.metadata["kernel_component"] = random.choice(["Memory Manager", "Scheduler", "File System"])
                zero_day_doc.metadata["privilege_escalation"] = True
            elif "Container" in attack_type:
                zero_day_doc.metadata["container_technology"] = random.choice(["Docker", "Kubernetes", "LXC"])
                zero_day_doc.metadata["escape_vector"] = random.choice(["Volume Mount", "Privileged Mode", "Namespace Escape"])
            elif "Hypervisor" in attack_type:
                zero_day_doc.metadata["hypervisor_type"] = random.choice(["KVM", "Xen", "VMware"])
                zero_day_doc.metadata["vm_escape"] = True
            elif "Microarchitectural" in attack_type:
                zero_day_doc.metadata["cpu_vulnerability"] = random.choice(["Cache Timing", "Speculative Execution", "Out-of-Order Execution"])
                zero_day_doc.metadata["side_channel"] = True
            elif "Encrypted" in attack_type:
                zero_day_doc.metadata["encryption_protocol"] = random.choice(["TLS", "SSH", "VPN"])
                zero_day_doc.metadata["exploitation_technique"] = random.choice(["Padding Oracle", "Downgrade Attack", "Implementation Flaw"])
            elif "IoT" in attack_type:
                zero_day_doc.metadata["device_type"] = random.choice(["Camera", "Thermostat", "Smart Speaker"])
                zero_day_doc.metadata["botnet_activity"] = random.choice(["Recruitment", "Command and Control", "Attack"])
            
            # Add to list
            zero_day_docs.append(zero_day_doc)
        
        # Add some benign documents
        num_benign = min(self.num_queries, len(documents) // 10)
        benign_docs = [doc for doc in documents if not doc.metadata.get("is_attack", False)]
        
        if len(benign_docs) >= num_benign:
            selected_benign = random.sample(benign_docs, num_benign)
            
            for i, doc in enumerate(selected_benign):
                # Create a copy of the document
                benign_doc = Document(
                    id=f"benign_{i}",
                    content=doc.content,
                    metadata=doc.metadata.copy()
                )
                
                zero_day_docs.append(benign_doc)
        
        logger.info(f"Created {len(zero_day_docs)} documents for zero-day scenario")
        return zero_day_docs
    
    def _generate_queries(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate queries for zero-day attacks.
        
        Args:
            documents: List of documents with zero-day attacks
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Group documents by attack type
        attack_groups = {}
        benign_docs = []
        
        for doc in documents:
            if doc.metadata.get("is_attack", False) and doc.metadata.get("zero_day", False):
                attack_type = doc.metadata.get("attack_type", "unknown")
                if attack_type not in attack_groups:
                    attack_groups[attack_type] = []
                attack_groups[attack_type].append(doc)
            else:
                benign_docs.append(doc)
        
        # Generate queries for each attack type
        for attack_type, attack_docs in attack_groups.items():
            # Determine number of queries for this attack type
            num_queries = max(1, int(self.num_queries * 0.8 * (len(attack_docs) / len(documents))))
            
            # Generate queries
            for i in range(num_queries):
                # Select a random document
                doc = random.choice(attack_docs)
                
                # Generate query text
                query_text = self._generate_query_text(doc, attack_type)
                
                # Create query
                query = {
                    "id": f"{attack_type.replace(' ', '_').lower()}_{i}",
                    "text": query_text,
                    "type": "zero_day",
                    "attack_type": attack_type,
                    "document_id": doc.id
                }
                
                queries.append(query)
        
        # Generate queries for benign traffic
        num_benign_queries = self.num_queries - len(queries)
        
        for i in range(num_benign_queries):
            if benign_docs:
                # Select a random document
                doc = random.choice(benign_docs)
                
                # Generate query text
                query_text = self._generate_benign_query_text(doc)
                
                # Create query
                query = {
                    "id": f"benign_{i}",
                    "text": query_text,
                    "type": "benign",
                    "document_id": doc.id
                }
                
                queries.append(query)
        
        # Shuffle queries
        random.shuffle(queries)
        
        logger.info(f"Generated {len(queries)} queries for zero-day scenario")
        return queries
    
    def _generate_query_text(self, document: Document, attack_type: str) -> str:
        """
        Generate query text for a zero-day attack.
        
        Args:
            document: Document with zero-day attack
            attack_type: Type of zero-day attack
            
        Returns:
            Generated query text
        """
        # Extract metadata
        src_ip = document.metadata.get("src_ip", "")
        dst_ip = document.metadata.get("dst_ip", "")
        src_port = document.metadata.get("src_port", "")
        dst_port = document.metadata.get("dst_port", "")
        protocol = document.metadata.get("protocol", "")
        
        # Generate query templates based on attack type
        templates = [
            f"Investigate unusual traffic from {src_ip} to {dst_ip}",
            f"Analyze suspicious {protocol} packets to {dst_ip}",
            f"Check for anomalies in traffic to port {dst_port}",
            f"Examine potential security threats from {src_ip}",
            f"Investigate unexpected behavior in {protocol} traffic"
        ]
        
        # Add attack-specific templates
        if "Protocol" in attack_type:
            templates.extend([
                f"Analyze unusual protocol behavior in traffic from {src_ip}",
                f"Investigate protocol anomalies to {dst_ip}:{dst_port}",
                f"Check for protocol exploitation attempts in {protocol} traffic"
            ])
        elif "Firmware" in attack_type:
            templates.extend([
                f"Investigate potential firmware exploitation from {src_ip}",
                f"Analyze traffic targeting device firmware at {dst_ip}",
                f"Check for firmware vulnerability exploitation attempts"
            ])
        elif "Supply Chain" in attack_type:
            templates.extend([
                f"Analyze potential supply chain compromise involving {dst_ip}",
                f"Investigate suspicious library calls to {dst_ip}:{dst_port}",
                f"Check for signs of compromised components in traffic"
            ])
        elif "API" in attack_type:
            templates.extend([
                f"Investigate unusual API calls to {dst_ip}",
                f"Analyze suspicious API traffic on port {dst_port}",
                f"Check for API manipulation attempts from {src_ip}"
            ])
        elif "Kernel" in attack_type:
            templates.extend([
                f"Investigate potential kernel exploitation from {src_ip}",
                f"Analyze traffic patterns suggesting kernel vulnerabilities",
                f"Check for privilege escalation attempts to {dst_ip}"
            ])
        elif "Container" in attack_type:
            templates.extend([
                f"Investigate container security issues involving {dst_ip}",
                f"Analyze potential container escape attempts from {src_ip}",
                f"Check for container boundary violations in traffic"
            ])
        elif "Hypervisor" in attack_type:
            templates.extend([
                f"Investigate virtualization security issues from {src_ip}",
                f"Analyze potential hypervisor attacks targeting {dst_ip}",
                f"Check for VM escape attempts in traffic"
            ])
        elif "Microarchitectural" in attack_type:
            templates.extend([
                f"Investigate potential side-channel attacks from {src_ip}",
                f"Analyze traffic patterns suggesting CPU vulnerabilities",
                f"Check for microarchitectural attack signatures"
            ])
        elif "Encrypted" in attack_type:
            templates.extend([
                f"Investigate encrypted traffic anomalies to {dst_ip}",
                f"Analyze potential TLS/SSL exploitation from {src_ip}",
                f"Check for encrypted protocol vulnerabilities"
            ])
        elif "IoT" in attack_type:
            templates.extend([
                f"Investigate IoT device communication from {src_ip}",
                f"Analyze potential IoT botnet activity involving {dst_ip}",
                f"Check for compromised IoT devices in network traffic"
            ])
        
        # Select a random template
        query_text = random.choice(templates)
        
        return query_text
    
    def _generate_benign_query_text(self, document: Document) -> str:
        """
        Generate query text for benign traffic.
        
        Args:
            document: Document with benign traffic
            
        Returns:
            Generated query text
        """
        # Extract metadata
        src_ip = document.metadata.get("src_ip", "")
        dst_ip = document.metadata.get("dst_ip", "")
        src_port = document.metadata.get("src_port", "")
        dst_port = document.metadata.get("dst_port", "")
        protocol = document.metadata.get("protocol", "")
        
        # Generate query templates
        templates = [
            f"Show traffic between {src_ip} and {dst_ip}",
            f"Display {protocol} connections to {dst_ip}",
            f"List network flows involving {src_ip}",
            f"Show packets to port {dst_port}",
            f"Display traffic summary for {src_ip}",
            f"List recent connections to {dst_ip}",
            f"Show {protocol} traffic statistics",
            f"Summarize network activity for {src_ip}",
            f"Display connection details for {dst_ip}:{dst_port}",
            f"Show traffic patterns between {src_ip} and {dst_ip}"
        ]
        
        # Select a random template
        query_text = random.choice(templates)
        
        return query_text
    
    def _generate_ground_truth(
        self,
        queries: List[Dict[str, Any]],
        documents: List[Document]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate ground truth for zero-day queries.
        
        Args:
            queries: List of queries
            documents: List of documents
            
        Returns:
            Dictionary mapping query IDs to lists of relevant document IDs
        """
        ground_truth = {}
        
        # Create document lookup
        doc_lookup = {doc.id: doc for doc in documents}
        
        # Process each query
        for query in queries:
            query_id = query["id"]
            document_id = query.get("document_id")
            
            # Start with the document that generated the query
            relevant_docs = []
            
            if document_id and document_id in doc_lookup:
                relevant_docs.append(doc_lookup[document_id])
            
            # Find additional relevant documents
            if query["type"] == "zero_day" and "attack_type" in query:
                attack_type = query["attack_type"]
                
                # Find documents with the same attack type
                for doc in documents:
                    if doc.id != document_id and doc.metadata.get("attack_type") == attack_type:
                        relevant_docs.append(doc)
            
            # Limit number of relevant documents
            if len(relevant_docs) > 10:
                # Keep the original document and sample from the rest
                original_doc = relevant_docs[0]
                other_docs = relevant_docs[1:]
                sampled_docs = random.sample(other_docs, 9)
                relevant_docs = [original_doc] + sampled_docs
            
            # Convert to dictionaries
            ground_truth[query_id] = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]
        
        logger.info(f"Generated ground truth for {len(queries)} zero-day queries")
        return ground_truth


class HighThroughputScenarioGenerator(TestScenarioGenerator):
    """
    Generator for high-throughput test scenarios.
    """
    def __init__(
        self,
        num_queries: int = 100,
        num_documents: int = 10000,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the high-throughput scenario generator.
        
        Args:
            num_queries: Number of queries to generate
            num_documents: Number of documents to generate
            output_dir: Directory to save generated scenarios
        """
        super().__init__("high_throughput", output_dir)
        self.num_queries = num_queries
        self.num_documents = num_documents
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate a high-throughput test scenario.
        
        Returns:
            Dictionary containing the generated scenario
        """
        # Check if scenario already exists
        existing_scenario = self.load()
        if existing_scenario:
            return existing_scenario
        
        logger.info(f"Generating high-throughput scenario with {self.num_documents} documents")
        
        # Generate synthetic documents
        documents = self._generate_documents()
        
        # Generate queries
        queries = self._generate_queries(documents)
        
        # Generate ground truth
        ground_truth = self._generate_ground_truth(queries, documents)
        
        # Create scenario
        scenario = {
            "name": self.scenario_name,
            "num_queries": len(queries),
            "num_documents": len(documents),
            "queries": queries,
            "ground_truth": ground_truth
        }
        
        # Save scenario
        self.save(scenario)
        
        return scenario
    
    def _generate_documents(self) -> List[Document]:
        """
        Generate synthetic documents for high-throughput testing.
        
        Returns:
            List of synthetic documents
        """
        documents = []
        
        # Generate IP ranges
        num_subnets = 100
        subnets = []
        
        for i in range(num_subnets):
            subnet = f"192.168.{i}.0/24"
            subnets.append(subnet)
        
        # Generate documents
        for i in range(self.num_documents):
            # Select a random subnet
            subnet_idx = i % num_subnets
            subnet = subnets[subnet_idx]
            subnet_prefix = subnet.split('/')[0].rsplit('.', 1)[0]
            
            # Generate IPs
            src_ip = f"{subnet_prefix}.{random.randint(1, 254)}"
            dst_ip = f"{subnet_prefix}.{random.randint(1, 254)}"
            
            # Ensure src_ip != dst_ip
            while src_ip == dst_ip:
                dst_ip = f"{subnet_prefix}.{random.randint(1, 254)}"
            
            # Generate ports
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 53, 25, 8080, 3389, 5900])
            
            # Generate protocol
            protocol = random.choice(['TCP', 'UDP', 'ICMP'])
            
            # Generate timestamp
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 60))
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Determine if this is an attack
            is_attack = random.random() < 0.2  # 20% chance of being an attack
            
            # Create document
            doc = {
                "id": f"doc_{i}",
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "timestamp": timestamp_str,
                "duration": random.random() * 10,
                "bytes_sent": random.randint(100, 10000),
                "bytes_received": random.randint(100, 10000),
                "packets_sent": random.randint(1, 100),
                "packets_received": random.randint(1, 100),
                "is_attack": is_attack
            }
            
            # Add attack-specific information
            if is_attack:
                attack_types = ['Port Scan', 'DoS', 'Brute Force', 'Data Exfiltration', 'Command and Control']
                attack_type = random.choice(attack_types)
                doc["attack_type"] = attack_type
                doc["threat_indicators"] = [f"Attack type: {attack_type}"]
                
                # Add attack-specific fields
                if attack_type == 'Port Scan':
                    doc["scan_type"] = random.choice(['SYN', 'FIN', 'XMAS', 'NULL'])
                elif attack_type == 'DoS':
                    doc["dos_method"] = random.choice(['SYN Flood', 'HTTP Flood', 'UDP Flood'])
                elif attack_type == 'Brute Force':
                    doc["service"] = random.choice(['SSH', 'FTP', 'HTTP'])
                    doc["attempts"] = random.randint(10, 100)
                elif attack_type == 'Data Exfiltration':
                    doc["data_type"] = random.choice(['PII', 'Financial', 'Intellectual Property'])
                    doc["volume"] = random.randint(1, 100)
                elif attack_type == 'Command and Control':
                    doc["c2_protocol"] = random.choice(['HTTP', 'DNS', 'ICMP'])
                    doc["beacon_interval"] = random.randint(60, 3600)
            
            # Format content
            content = f"""
Source IP: {doc['src_ip']}
Destination IP: {doc['dst_ip']}
Source Port: {doc['src_port']}
Destination Port: {doc['dst_port']}
Protocol: {doc['protocol']}
Timestamp: {doc['timestamp']}
Duration: {doc['duration']:.2f} seconds
Bytes Sent: {doc['bytes_sent']}
Bytes Received: {doc['bytes_received']}
Packets Sent: {doc['packets_sent']}
Packets Received: {doc['packets_received']}
"""
            
            if is_attack:
                content += f"""
Attack Type: {doc['attack_type']}
Threat Indicators: {', '.join(doc['threat_indicators'])}
"""
            
            # Create Document object
            document = Document(
                id=doc["id"],
                content=content,
                metadata=doc
            )
            
            documents.append(document)
        
        logger.info(f"Generated {len(documents)} synthetic documents")
        return documents
    
    def _generate_queries(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate queries for high-throughput testing.
        
        Args:
            documents: List of documents
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Group documents by subnet
        subnet_groups = {}
        
        for doc in documents:
            src_ip = doc.metadata.get("src_ip", "")
            subnet = '.'.join(src_ip.split('.')[:3])
            
            if subnet not in subnet_groups:
                subnet_groups[subnet] = []
            
            subnet_groups[subnet].append(doc)
        
        # Generate subnet queries
        num_subnet_queries = self.num_queries // 2
        
        for i in range(num_subnet_queries):
            # Select a random subnet
            if subnet_groups:
                subnet = random.choice(list(subnet_groups.keys()))
                subnet_docs = subnet_groups[subnet]
                
                # Generate query text
                query_text = f"Analyze all traffic in subnet {subnet}.0/24"
                
                # Create query
                query = {
                    "id": f"subnet_{i}",
                    "text": query_text,
                    "type": "subnet",
                    "subnet": subnet
                }
                
                queries.append(query)
        
        # Generate attack queries
        attack_docs = [doc for doc in documents if doc.metadata.get("is_attack", False)]
        
        if attack_docs:
            # Group by attack type
            attack_groups = {}
            
            for doc in attack_docs:
                attack_type = doc.metadata.get("attack_type", "unknown")
                
                if attack_type not in attack_groups:
                    attack_groups[attack_type] = []
                
                attack_groups[attack_type].append(doc)
            
            # Generate queries for each attack type
            remaining_queries = self.num_queries - len(queries)
            queries_per_type = remaining_queries // len(attack_groups) if attack_groups else 0
            
            for attack_type, attack_docs in attack_groups.items():
                for i in range(queries_per_type):
                    # Generate query text
                    query_text = f"Find all {attack_type} attacks in the network"
                    
                    # Create query
                    query = {
                        "id": f"{attack_type.lower().replace(' ', '_')}_{i}",
                        "text": query_text,
                        "type": "attack",
                        "attack_type": attack_type
                    }
                    
                    queries.append(query)
        
        # Fill remaining queries with random document queries
        remaining_queries = self.num_queries - len(queries)
        
        for i in range(remaining_queries):
            # Select a random document
            doc = random.choice(documents)
            
            # Generate query text
            src_ip = doc.metadata.get("src_ip", "")
            dst_ip = doc.metadata.get("dst_ip", "")
            
            query_text = f"Analyze traffic between {src_ip} and {dst_ip}"
            
            # Create query
            query = {
                "id": f"flow_{i}",
                "text": query_text,
                "type": "flow",
                "document_id": doc.id
            }
            
            queries.append(query)
        
        # Shuffle queries
        random.shuffle(queries)
        
        logger.info(f"Generated {len(queries)} queries for high-throughput scenario")
        return queries
    
    def _generate_ground_truth(
        self,
        queries: List[Dict[str, Any]],
        documents: List[Document]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate ground truth for high-throughput queries.
        
        Args:
            queries: List of queries
            documents: List of documents
            
        Returns:
            Dictionary mapping query IDs to lists of relevant document IDs
        """
        ground_truth = {}
        
        # Create document lookup
        doc_lookup = {doc.id: doc for doc in documents}
        
        # Process each query
        for query in queries:
            query_id = query["id"]
            query_type = query.get("type", "")
            
            relevant_docs = []
            
            if query_type == "subnet":
                # Find documents in the subnet
                subnet = query.get("subnet", "")
                
                if subnet:
                    for doc in documents:
                        src_ip = doc.metadata.get("src_ip", "")
                        dst_ip = doc.metadata.get("dst_ip", "")
                        
                        if src_ip.startswith(subnet) or dst_ip.startswith(subnet):
                            relevant_docs.append(doc)
            
            elif query_type == "attack":
                # Find documents with the attack type
                attack_type = query.get("attack_type", "")
                
                if attack_type:
                    for doc in documents:
                        if doc.metadata.get("attack_type") == attack_type:
                            relevant_docs.append(doc)
            
            elif query_type == "flow":
                # Find the specific document and related flows
                document_id = query.get("document_id", "")
                
                if document_id and document_id in doc_lookup:
                    doc = doc_lookup[document_id]
                    relevant_docs.append(doc)
                    
                    # Find related flows
                    src_ip = doc.metadata.get("src_ip", "")
                    dst_ip = doc.metadata.get("dst_ip", "")
                    
                    for other_doc in documents:
                        if other_doc.id != document_id:
                            other_src_ip = other_doc.metadata.get("src_ip", "")
                            other_dst_ip = other_doc.metadata.get("dst_ip", "")
                            
                            if (src_ip == other_src_ip and dst_ip == other_dst_ip) or \
                               (src_ip == other_dst_ip and dst_ip == other_src_ip):
                                relevant_docs.append(other_doc)
            
            # Limit number of relevant documents for performance
            if len(relevant_docs) > 100:
                relevant_docs = random.sample(relevant_docs, 100)
            
            # Convert to dictionaries
            ground_truth[query_id] = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]
        
        logger.info(f"Generated ground truth for {len(queries)} high-throughput queries")
        return ground_truth


class AdversarialScenarioGenerator(TestScenarioGenerator):
    """
    Generator for adversarial test scenarios.
    """
    def __init__(
        self,
        num_queries: int = 50,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the adversarial scenario generator.
        
        Args:
            num_queries: Number of queries to generate
            output_dir: Directory to save generated scenarios
        """
        super().__init__("adversarial", output_dir)
        self.num_queries = num_queries
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate an adversarial test scenario.
        
        Returns:
            Dictionary containing the generated scenario
        """
        # Check if scenario already exists
        existing_scenario = self.load()
        if existing_scenario:
            return existing_scenario
        
        # Load datasets
        cic_processor = CICIDSProcessor()
        unsw_processor = UNSWProcessor()
        
        # Process datasets
        cic_splits = cic_processor.process()
        unsw_splits = unsw_processor.process()
        
        # Use test splits for evaluation
        cic_test_docs = cic_splits.get("test", [])
        unsw_test_docs = unsw_splits.get("test", [])
        
        # Combine documents
        all_docs = cic_test_docs + unsw_test_docs
        
        if not all_docs:
            logger.error("No documents found for adversarial scenario")
            raise ValueError("No documents found for adversarial scenario")
        
        logger.info(f"Generating adversarial scenario with {len(all_docs)} documents")
        
        # Create adversarial examples
        adversarial_docs = self._create_adversarial_examples(all_docs)
        
        # Generate queries
        queries = self._generate_queries(adversarial_docs)
        
        # Generate ground truth
        ground_truth = self._generate_ground_truth(queries, adversarial_docs)
        
        # Create scenario
        scenario = {
            "name": self.scenario_name,
            "num_queries": len(queries),
            "num_documents": len(adversarial_docs),
            "queries": queries,
            "ground_truth": ground_truth
        }
        
        # Save scenario
        self.save(scenario)
        
        return scenario
    
    def _create_adversarial_examples(self, documents: List[Document]) -> List[Document]:
        """
        Create adversarial examples.
        
        Args:
            documents: List of documents to base adversarial examples on
            
        Returns:
            List of documents with adversarial examples
        """
        # Define adversarial techniques
        adversarial_techniques = [
            "Obfuscation",
            "Fragmentation",
            "Encryption",
            "Tunneling",
            "Protocol Violation",
            "Timing Manipulation",
            "Mimicry",
            "Polymorphism",
            "Traffic Manipulation",
            "Evasion"
        ]
        
        # Create adversarial examples
        adversarial_docs = []
        
        # Select attack documents to modify
        attack_docs = [doc for doc in documents if doc.metadata.get("is_attack", False)]
        
        if not attack_docs:
            # If no attack documents, use benign documents
            attack_docs = documents
        
        # Select a subset of documents to modify
        num_adversarial = min(self.num_queries * 2, len(attack_docs))
        selected_docs = random.sample(attack_docs, num_adversarial)
        
        for i, doc in enumerate(selected_docs):
            # Create a copy of the document
            adversarial_doc = Document(
                id=f"adversarial_{i}",
                content=doc.content,
                metadata=doc.metadata.copy()
            )
            
            # Assign an adversarial technique
            technique = random.choice(adversarial_techniques)
            
            # Modify metadata
            adversarial_doc.metadata["adversarial"] = True
            adversarial_doc.metadata["adversarial_technique"] = technique
            
            # Add adversarial-specific metadata
            if technique == "Obfuscation":
                adversarial_doc.metadata["obfuscation_method"] = random.choice(["Padding", "Junk Data", "Encoding"])
                adversarial_doc.metadata["detection_difficulty"] = random.uniform(0.7, 1.0)
            elif technique == "Fragmentation":
                adversarial_doc.metadata["fragment_size"] = random.randint(8, 64)
                adversarial_doc.metadata["fragment_count"] = random.randint(10, 100)
            elif technique == "Encryption":
                adversarial_doc.metadata["encryption_type"] = random.choice(["Custom", "Standard", "Layered"])
                adversarial_doc.metadata["key_rotation"] = random.choice([True, False])
            elif technique == "Tunneling":
                adversarial_doc.metadata["tunnel_protocol"] = random.choice(["DNS", "ICMP", "HTTP"])
                adversarial_doc.metadata["tunnel_depth"] = random.randint(1, 3)
            elif technique == "Protocol Violation":
                adversarial_doc.metadata["violation_type"] = random.choice(["Header Manipulation", "Invalid Sequence", "Malformed Packet"])
                adversarial_doc.metadata["violation_severity"] = random.uniform(0.5, 1.0)
            elif technique == "Timing Manipulation":
                adversarial_doc.metadata["timing_pattern"] = random.choice(["Random", "Periodic", "Adaptive"])
                adversarial_doc.metadata["interval_variation"] = random.uniform(0.1, 0.9)
            elif technique == "Mimicry":
                adversarial_doc.metadata["mimicked_traffic"] = random.choice(["HTTP", "DNS", "HTTPS", "NTP"])
                adversarial_doc.metadata["similarity_score"] = random.uniform(0.8, 0.99)
            elif technique == "Polymorphism":
                adversarial_doc.metadata["mutation_rate"] = random.uniform(0.1, 0.5)
                adversarial_doc.metadata["variant_id"] = random.randint(1, 100)
            elif technique == "Traffic Manipulation":
                adversarial_doc.metadata["manipulation_type"] = random.choice(["Reordering", "Duplication", "Insertion"])
                adversarial_doc.metadata["manipulation_extent"] = random.uniform(0.2, 0.8)
            elif technique == "Evasion":
                adversarial_doc.metadata["evasion_target"] = random.choice(["IDS", "Firewall", "DPI"])
                adversarial_doc.metadata["evasion_method"] = random.choice(["Slow Rate", "Split Payload", "TTL Manipulation"])
            
            # Add to list
            adversarial_docs.append(adversarial_doc)
        
        # Add some benign documents
        num_benign = min(self.num_queries, len(documents) // 10)
        benign_docs = [doc for doc in documents if not doc.metadata.get("is_attack", False)]
        
        if len(benign_docs) >= num_benign:
            selected_benign = random.sample(benign_docs, num_benign)
            
            for i, doc in enumerate(selected_benign):
                # Create a copy of the document
                benign_doc = Document(
                    id=f"benign_{i}",
                    content=doc.content,
                    metadata=doc.metadata.copy()
                )
                
                adversarial_docs.append(benign_doc)
        
        logger.info(f"Created {len(adversarial_docs)} documents for adversarial scenario")
        return adversarial_docs
    
    def _generate_queries(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate queries for adversarial testing.
        
        Args:
            documents: List of documents with adversarial examples
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Group documents by adversarial technique
        technique_groups = {}
        benign_docs = []
        
        for doc in documents:
            if doc.metadata.get("adversarial", False):
                technique = doc.metadata.get("adversarial_technique", "unknown")
                if technique not in technique_groups:
                    technique_groups[technique] = []
                technique_groups[technique].append(doc)
            else:
                benign_docs.append(doc)
        
        # Generate queries for each adversarial technique
        for technique, technique_docs in technique_groups.items():
            # Determine number of queries for this technique
            num_queries = max(1, int(self.num_queries * 0.8 * (len(technique_docs) / len(documents))))
            
            # Generate queries
            for i in range(num_queries):
                # Select a random document
                doc = random.choice(technique_docs)
                
                # Generate query text
                query_text = self._generate_query_text(doc, technique)
                
                # Create query
                query = {
                    "id": f"{technique.lower().replace(' ', '_')}_{i}",
                    "text": query_text,
                    "type": "adversarial",
                    "technique": technique,
                    "document_id": doc.id
                }
                
                queries.append(query)
        
        # Generate queries for benign traffic
        num_benign_queries = self.num_queries - len(queries)
        
        for i in range(num_benign_queries):
            if benign_docs:
                # Select a random document
                doc = random.choice(benign_docs)
                
                # Generate query text
                query_text = self._generate_benign_query_text(doc)
                
                # Create query
                query = {
                    "id": f"benign_{i}",
                    "text": query_text,
                    "type": "benign",
                    "document_id": doc.id
                }
                
                queries.append(query)
        
        # Shuffle queries
        random.shuffle(queries)
        
        logger.info(f"Generated {len(queries)} queries for adversarial scenario")
        return queries
    
    def _generate_query_text(self, document: Document, technique: str) -> str:
        """
        Generate query text for an adversarial example.
        
        Args:
            document: Document with adversarial example
            technique: Adversarial technique
            
        Returns:
            Generated query text
        """
        # Extract metadata
        src_ip = document.metadata.get("src_ip", "")
        dst_ip = document.metadata.get("dst_ip", "")
        src_port = document.metadata.get("src_port", "")
        dst_port = document.metadata.get("dst_port", "")
        protocol = document.metadata.get("protocol", "")
        
        # Generate query templates based on technique
        templates = [
            f"Investigate evasive traffic from {src_ip} to {dst_ip}",
            f"Analyze potential adversarial behavior in {protocol} traffic",
            f"Check for attack evasion techniques in traffic to {dst_ip}:{dst_port}",
            f"Examine suspicious traffic patterns from {src_ip}"
        ]
        
        # Add technique-specific templates
        if technique == "Obfuscation":
            templates.extend([
                f"Investigate obfuscated traffic from {src_ip}",
                f"Analyze potentially hidden attacks in traffic to {dst_ip}",
                f"Check for obfuscation techniques in {protocol} traffic"
            ])
        elif technique == "Fragmentation":
            templates.extend([
                f"Investigate fragmented packets from {src_ip}",
                f"Analyze traffic with unusual fragmentation to {dst_ip}",
                f"Check for fragmentation-based evasion in {protocol} traffic"
            ])
        elif technique == "Encryption":
            templates.extend([
                f"Investigate encrypted traffic from {src_ip}",
                f"Analyze suspicious encrypted connections to {dst_ip}:{dst_port}",
                f"Check for encryption-based evasion in traffic"
            ])
        elif technique == "Tunneling":
            templates.extend([
                f"Investigate potential tunneling from {src_ip}",
                f"Analyze traffic that may contain tunneled protocols to {dst_ip}",
                f"Check for protocol tunneling in {protocol} traffic"
            ])
        elif technique == "Protocol Violation":
            templates.extend([
                f"Investigate protocol violations from {src_ip}",
                f"Analyze malformed {protocol} traffic to {dst_ip}",
                f"Check for protocol manipulation in traffic"
            ])
        elif technique == "Timing Manipulation":
            templates.extend([
                f"Investigate timing anomalies in traffic from {src_ip}",
                f"Analyze traffic with unusual timing patterns to {dst_ip}",
                f"Check for timing-based evasion techniques"
            ])
        elif technique == "Mimicry":
            templates.extend([
                f"Investigate traffic mimicking legitimate protocols from {src_ip}",
                f"Analyze potential mimicry attacks to {dst_ip}",
                f"Check for attacks disguised as legitimate traffic"
            ])
        elif technique == "Polymorphism":
            templates.extend([
                f"Investigate polymorphic behavior in traffic from {src_ip}",
                f"Analyze traffic with changing patterns to {dst_ip}",
                f"Check for polymorphic attack techniques"
            ])
        elif technique == "Traffic Manipulation":
            templates.extend([
                f"Investigate manipulated traffic from {src_ip}",
                f"Analyze potentially altered packets to {dst_ip}",
                f"Check for traffic manipulation techniques"
            ])
        elif technique == "Evasion":
            templates.extend([
                f"Investigate IDS evasion attempts from {src_ip}",
                f"Analyze traffic designed to evade detection to {dst_ip}",
                f"Check for security evasion techniques in {protocol} traffic"
            ])
        
        # Select a random template
        query_text = random.choice(templates)
        
        return query_text
    
    def _generate_benign_query_text(self, document: Document) -> str:
        """
        Generate query text for benign traffic.
        
        Args:
            document: Document with benign traffic
            
        Returns:
            Generated query text
        """
        # Extract metadata
        src_ip = document.metadata.get("src_ip", "")
        dst_ip = document.metadata.get("dst_ip", "")
        src_port = document.metadata.get("src_port", "")
        dst_port = document.metadata.get("dst_port", "")
        protocol = document.metadata.get("protocol", "")
        
        # Generate query templates
        templates = [
            f"Show traffic between {src_ip} and {dst_ip}",
            f"Display {protocol} connections to {dst_ip}",
            f"List network flows involving {src_ip}",
            f"Show packets to port {dst_port}",
            f"Display traffic summary for {src_ip}",
            f"List recent connections to {dst_ip}",
            f"Show {protocol} traffic statistics",
            f"Summarize network activity for {src_ip}",
            f"Display connection details for {dst_ip}:{dst_port}",
            f"Show traffic patterns between {src_ip} and {dst_ip}"
        ]
        
        # Select a random template
        query_text = random.choice(templates)
        
        return query_text
    
    def _generate_ground_truth(
        self,
        queries: List[Dict[str, Any]],
        documents: List[Document]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate ground truth for adversarial queries.
        
        Args:
            queries: List of queries
            documents: List of documents
            
        Returns:
            Dictionary mapping query IDs to lists of relevant document IDs
        """
        ground_truth = {}
        
        # Create document lookup
        doc_lookup = {doc.id: doc for doc in documents}
        
        # Process each query
        for query in queries:
            query_id = query["id"]
            document_id = query.get("document_id")
            
            # Start with the document that generated the query
            relevant_docs = []
            
            if document_id and document_id in doc_lookup:
                relevant_docs.append(doc_lookup[document_id])
            
            # Find additional relevant documents
            if query["type"] == "adversarial" and "technique" in query:
                technique = query["technique"]
                
                # Find documents with the same technique
                for doc in documents:
                    if doc.id != document_id and doc.metadata.get("adversarial_technique") == technique:
                        relevant_docs.append(doc)
            
            # Limit number of relevant documents
            if len(relevant_docs) > 10:
                # Keep the original document and sample from the rest
                original_doc = relevant_docs[0]
                other_docs = relevant_docs[1:]
                sampled_docs = random.sample(other_docs, 9)
                relevant_docs = [original_doc] + sampled_docs
            
            # Convert to dictionaries
            ground_truth[query_id] = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ]
        
        logger.info(f"Generated ground truth for {len(queries)} adversarial queries")
        return ground_truth


class BenchmarkingFramework:
    """
    Framework for benchmarking RAG systems.
    """
    def __init__(
        self,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the benchmarking framework.
        
        Args:
            output_dir: Directory to save benchmarking results
        """
        self.output_dir = output_dir or Path("/home/ubuntu/research/experiments/benchmarks")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scenarios = {}
    
    def load_scenarios(self) -> None:
        """Load all test scenarios."""
        # Standard scenarios
        for dataset in ["cic_ids2017", "unsw_nb15", "custom_pcap"]:
            generator = StandardScenarioGenerator(dataset)
            scenario = generator.generate()
            self.scenarios[f"standard_{dataset}"] = scenario
        
        # Zero-day scenario
        zero_day_generator = ZeroDayScenarioGenerator()
        zero_day_scenario = zero_day_generator.generate()
        self.scenarios["zero_day"] = zero_day_scenario
        
        # High-throughput scenario
        high_throughput_generator = HighThroughputScenarioGenerator()
        high_throughput_scenario = high_throughput_generator.generate()
        self.scenarios["high_throughput"] = high_throughput_scenario
        
        # Adversarial scenario
        adversarial_generator = AdversarialScenarioGenerator()
        adversarial_scenario = adversarial_generator.generate()
        self.scenarios["adversarial"] = adversarial_scenario
        
        logger.info(f"Loaded {len(self.scenarios)} test scenarios")
    
    def prepare_queries(self, scenario_name: str) -> Tuple[List[Query], Dict[str, List[Document]]]:
        """
        Prepare queries and ground truth for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Tuple of (queries, ground_truth)
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        
        # Convert query dictionaries to Query objects
        queries = []
        
        for query_dict in scenario["queries"]:
            query = Query(
                id=query_dict["id"],
                text=query_dict["text"]
            )
            
            queries.append(query)
        
        # Convert ground truth dictionaries to Document objects
        ground_truth = {}
        
        for query_id, doc_dicts in scenario["ground_truth"].items():
            ground_truth[query_id] = []
            
            for doc_dict in doc_dicts:
                doc = Document(
                    id=doc_dict["id"],
                    content=doc_dict["content"],
                    metadata=doc_dict["metadata"]
                )
                
                ground_truth[query_id].append(doc)
        
        return queries, ground_truth
    
    def save_benchmark_config(
        self,
        system_names: List[str],
        scenario_names: List[str]
    ) -> None:
        """
        Save benchmark configuration.
        
        Args:
            system_names: List of system names to benchmark
            scenario_names: List of scenario names to benchmark
        """
        config = {
            "systems": system_names,
            "scenarios": scenario_names,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save configuration
        config_file = os.path.join(self.output_dir, "benchmark_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved benchmark configuration to {config_file}")
    
    def load_benchmark_config(self) -> Optional[Dict[str, Any]]:
        """
        Load benchmark configuration.
        
        Returns:
            Benchmark configuration or None if not found
        """
        config_file = os.path.join(self.output_dir, "benchmark_config.json")
        
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            
            logger.info(f"Loaded benchmark configuration from {config_file}")
            return config
        
        logger.warning(f"Benchmark configuration not found: {config_file}")
        return None
