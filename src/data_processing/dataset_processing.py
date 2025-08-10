"""
Dataset processing utilities for network security packet analysis.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import zipfile
import gzip
import tarfile
import json
import csv
from datetime import datetime
import random
from pathlib import Path

from ..common.base import Document
from ..common.config import DATASET_CONFIG
from ..utils.network_utils import format_packet_data, generate_flow_id, extract_protocol_features

# Configure logging
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """
    Base class for dataset processors.
    """
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        Initialize the dataset processor.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset files
            output_path: Path to save processed data
        """
        self.dataset_name = dataset_name
        
        # Get configuration
        if dataset_name in DATASET_CONFIG:
            config = DATASET_CONFIG[dataset_name]
            self.dataset_path = dataset_path or config["path"]
            self.train_ratio = config["train_ratio"]
            self.val_ratio = config["val_ratio"]
            self.test_ratio = config["test_ratio"]
        else:
            self.dataset_path = dataset_path or Path(f"/home/ubuntu/research/data/{dataset_name}")
            self.train_ratio = 0.7
            self.val_ratio = 0.15
            self.test_ratio = 0.15
        
        # Set output path
        self.output_path = output_path or Path(f"/home/ubuntu/research/data/processed/{dataset_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
    
    def process(self) -> Dict[str, List[Document]]:
        """
        Process the dataset.
        
        Returns:
            Dictionary mapping split names to lists of documents
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def load_raw_data(self) -> Any:
        """
        Load raw data from the dataset.
        
        Returns:
            Raw data in dataset-specific format
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def preprocess_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Preprocess raw data into a standardized format.
        
        Args:
            raw_data: Raw data from load_raw_data
            
        Returns:
            List of preprocessed data records
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def convert_to_documents(self, preprocessed_data: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert preprocessed data to Document objects.
        
        Args:
            preprocessed_data: Preprocessed data from preprocess_data
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for record in preprocessed_data:
            # Generate ID if not present
            if "id" not in record:
                record["id"] = generate_flow_id(record)
            
            # Format content if not present
            if "content" not in record:
                record["content"] = format_packet_data(record)
            
            # Extract metadata
            metadata = record.get("metadata", {})
            if not metadata:
                metadata = {k: v for k, v in record.items() if k not in ["id", "content"]}
                
                # Extract protocol features
                protocol_features = extract_protocol_features(record)
                metadata.update(protocol_features)
            
            # Create document
            document = Document(
                id=record["id"],
                content=record["content"],
                metadata=metadata
            )
            
            documents.append(document)
        
        return documents
    
    def split_data(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary mapping split names to lists of documents
        """
        # Shuffle documents
        random.shuffle(documents)
        
        # Calculate split indices
        n_documents = len(documents)
        n_train = int(n_documents * self.train_ratio)
        n_val = int(n_documents * self.val_ratio)
        
        # Split documents
        train_docs = documents[:n_train]
        val_docs = documents[n_train:n_train + n_val]
        test_docs = documents[n_train + n_val:]
        
        return {
            "train": train_docs,
            "val": val_docs,
            "test": test_docs
        }
    
    def save_splits(self, splits: Dict[str, List[Document]]) -> None:
        """
        Save data splits to disk.
        
        Args:
            splits: Dictionary mapping split names to lists of documents
        """
        for split_name, documents in splits.items():
            # Convert documents to dictionaries
            docs_dicts = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            
            # Save to JSON file
            output_file = os.path.join(self.output_path, f"{split_name}.json")
            with open(output_file, "w") as f:
                json.dump(docs_dicts, f, indent=2)
            
            logger.info(f"Saved {len(documents)} documents to {output_file}")
    
    def load_splits(self) -> Dict[str, List[Document]]:
        """
        Load data splits from disk.
        
        Returns:
            Dictionary mapping split names to lists of documents
        """
        splits = {}
        
        for split_name in ["train", "val", "test"]:
            input_file = os.path.join(self.output_path, f"{split_name}.json")
            
            if os.path.exists(input_file):
                with open(input_file, "r") as f:
                    docs_dicts = json.load(f)
                
                documents = [
                    Document(
                        id=doc_dict["id"],
                        content=doc_dict["content"],
                        metadata=doc_dict["metadata"]
                    )
                    for doc_dict in docs_dicts
                ]
                
                splits[split_name] = documents
                logger.info(f"Loaded {len(documents)} documents from {input_file}")
            else:
                logger.warning(f"Split file not found: {input_file}")
                splits[split_name] = []
        
        return splits


class CICIDSProcessor(DatasetProcessor):
    """
    Processor for CIC-IDS2017 dataset.
    """
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        Initialize the CIC-IDS2017 processor.
        
        Args:
            dataset_path: Path to the dataset files
            output_path: Path to save processed data
        """
        super().__init__("cic_ids2017", dataset_path, output_path)
    
    def process(self) -> Dict[str, List[Document]]:
        """
        Process the CIC-IDS2017 dataset.
        
        Returns:
            Dictionary mapping split names to lists of documents
        """
        # Check if processed data already exists
        if os.path.exists(os.path.join(self.output_path, "train.json")):
            logger.info("Processed data already exists, loading from disk")
            return self.load_splits()
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(raw_data)
        
        # Convert to documents
        documents = self.convert_to_documents(preprocessed_data)
        
        # Split data
        splits = self.split_data(documents)
        
        # Save splits
        self.save_splits(splits)
        
        return splits
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from the CIC-IDS2017 dataset.
        
        Returns:
            DataFrame containing the dataset
        """
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # List CSV files in the dataset path
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.dataset_path}")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # Load and concatenate CSV files
        dataframes = []
        
        for csv_file in csv_files:
            file_path = os.path.join(self.dataset_path, csv_file)
            try:
                df = pd.read_csv(file_path, low_memory=False)
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {csv_file}")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if not dataframes:
            logger.warning("No data loaded from CSV files")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # Concatenate dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} records")
        
        return combined_df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create a sample dataset for demonstration purposes.
        
        Returns:
            DataFrame containing sample data
        """
        # Define column names based on CIC-IDS2017 format
        columns = [
            'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
            'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packets',
            'Total Backward Packets', 'Total Length of Fwd Packets',
            'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Bwd Packet Length Max',
            'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
            'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
            'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
            'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
            'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
            'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
            'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
            'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean',
            'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
            'Idle Max', 'Idle Min', 'Label'
        ]
        
        # Generate sample data
        n_samples = 1000
        data = []
        
        # Generate benign traffic
        for i in range(int(n_samples * 0.7)):
            src_ip = f"192.168.1.{random.randint(1, 254)}"
            dst_ip = f"10.0.0.{random.randint(1, 254)}"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 53, 25])
            protocol = random.choice(['TCP', 'UDP', 'ICMP'])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            row = [
                f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol}",  # Flow ID
                src_ip,  # Source IP
                src_port,  # Source Port
                dst_ip,  # Destination IP
                dst_port,  # Destination Port
                protocol,  # Protocol
                timestamp,  # Timestamp
                random.randint(1000, 10000),  # Flow Duration
            ]
            
            # Add remaining columns with random values
            row.extend([random.random() * 100 for _ in range(len(columns) - 8 - 1)])
            
            # Add label
            row.append('BENIGN')
            
            data.append(row)
        
        # Generate attack traffic
        attack_types = ['DoS Hulk', 'PortScan', 'DDoS', 'FTP-Patator', 'SSH-Patator']
        
        for i in range(int(n_samples * 0.3)):
            src_ip = f"192.168.1.{random.randint(1, 254)}"
            dst_ip = f"10.0.0.{random.randint(1, 254)}"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 53, 25])
            protocol = random.choice(['TCP', 'UDP', 'ICMP'])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            row = [
                f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol}",  # Flow ID
                src_ip,  # Source IP
                src_port,  # Source Port
                dst_ip,  # Destination IP
                dst_port,  # Destination Port
                protocol,  # Protocol
                timestamp,  # Timestamp
                random.randint(1000, 10000),  # Flow Duration
            ]
            
            # Add remaining columns with random values
            row.extend([random.random() * 100 for _ in range(len(columns) - 8 - 1)])
            
            # Add label
            row.append(random.choice(attack_types))
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Save sample data
        os.makedirs(self.dataset_path, exist_ok=True)
        sample_path = os.path.join(self.dataset_path, "sample_data.csv")
        df.to_csv(sample_path, index=False)
        logger.info(f"Created sample dataset with {len(df)} records at {sample_path}")
        
        return df
    
    def preprocess_data(self, raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Preprocess raw data into a standardized format.
        
        Args:
            raw_data: Raw data from load_raw_data
            
        Returns:
            List of preprocessed data records
        """
        preprocessed_data = []
        
        # Rename columns to standardized format
        column_mapping = {
            'Flow ID': 'flow_id',
            'Source IP': 'src_ip',
            'Source Port': 'src_port',
            'Destination IP': 'dst_ip',
            'Destination Port': 'dst_port',
            'Protocol': 'protocol',
            'Timestamp': 'timestamp',
            'Flow Duration': 'duration',
            'Total Fwd Packets': 'packets_sent',
            'Total Backward Packets': 'packets_received',
            'Total Length of Fwd Packets': 'bytes_sent',
            'Total Length of Bwd Packets': 'bytes_received',
            'Label': 'label'
        }
        
        # Check if columns exist in the DataFrame
        for old_col, new_col in column_mapping.items():
            if old_col not in raw_data.columns:
                logger.warning(f"Column {old_col} not found in dataset")
        
        # Process each row
        for _, row in raw_data.iterrows():
            record = {}
            
            # Add mapped columns
            for old_col, new_col in column_mapping.items():
                if old_col in row.index:
                    record[new_col] = row[old_col]
            
            # Add additional features
            record['threat_indicators'] = []
            
            # Check if this is an attack
            if 'label' in record and record['label'] != 'BENIGN':
                record['threat_indicators'].append(f"Attack type: {record['label']}")
                record['is_attack'] = True
            else:
                record['is_attack'] = False
            
            # Add to preprocessed data
            preprocessed_data.append(record)
        
        logger.info(f"Preprocessed {len(preprocessed_data)} records")
        return preprocessed_data


class UNSWProcessor(DatasetProcessor):
    """
    Processor for UNSW-NB15 dataset.
    """
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        Initialize the UNSW-NB15 processor.
        
        Args:
            dataset_path: Path to the dataset files
            output_path: Path to save processed data
        """
        super().__init__("unsw_nb15", dataset_path, output_path)
    
    def process(self) -> Dict[str, List[Document]]:
        """
        Process the UNSW-NB15 dataset.
        
        Returns:
            Dictionary mapping split names to lists of documents
        """
        # Check if processed data already exists
        if os.path.exists(os.path.join(self.output_path, "train.json")):
            logger.info("Processed data already exists, loading from disk")
            return self.load_splits()
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(raw_data)
        
        # Convert to documents
        documents = self.convert_to_documents(preprocessed_data)
        
        # Split data
        splits = self.split_data(documents)
        
        # Save splits
        self.save_splits(splits)
        
        return splits
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from the UNSW-NB15 dataset.
        
        Returns:
            DataFrame containing the dataset
        """
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # List CSV files in the dataset path
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.dataset_path}")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # Load and concatenate CSV files
        dataframes = []
        
        for csv_file in csv_files:
            file_path = os.path.join(self.dataset_path, csv_file)
            try:
                df = pd.read_csv(file_path, low_memory=False)
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {csv_file}")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if not dataframes:
            logger.warning("No data loaded from CSV files")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # Concatenate dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} records")
        
        return combined_df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create a sample dataset for demonstration purposes.
        
        Returns:
            DataFrame containing sample data
        """
        # Define column names based on UNSW-NB15 format
        columns = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
            'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
            'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
            'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
            'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
            'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
            'ct_dst_src_ltm', 'attack_cat', 'label'
        ]
        
        # Generate sample data
        n_samples = 1000
        data = []
        
        # Generate benign traffic
        for i in range(int(n_samples * 0.7)):
            src_ip = f"192.168.1.{random.randint(1, 254)}"
            dst_ip = f"10.0.0.{random.randint(1, 254)}"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 53, 25])
            protocol = random.choice(['tcp', 'udp', 'icmp'])
            service = random.choice(['http', 'dns', 'smtp', 'ssh', '-'])
            
            row = [
                src_ip,  # srcip
                src_port,  # sport
                dst_ip,  # dstip
                dst_port,  # dsport
                protocol,  # proto
                random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST']),  # state
                random.random() * 10,  # dur
                random.randint(100, 10000),  # sbytes
                random.randint(100, 10000),  # dbytes
            ]
            
            # Add remaining columns with random values
            row.extend([random.random() * 100 for _ in range(len(columns) - 9 - 2)])
            
            # Add attack category and label
            row.append('')  # attack_cat
            row.append(0)   # label
            
            data.append(row)
        
        # Generate attack traffic
        attack_types = ['Fuzzers', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 
                        'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
        
        for i in range(int(n_samples * 0.3)):
            src_ip = f"192.168.1.{random.randint(1, 254)}"
            dst_ip = f"10.0.0.{random.randint(1, 254)}"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 53, 25])
            protocol = random.choice(['tcp', 'udp', 'icmp'])
            service = random.choice(['http', 'dns', 'smtp', 'ssh', '-'])
            
            row = [
                src_ip,  # srcip
                src_port,  # sport
                dst_ip,  # dstip
                dst_port,  # dsport
                protocol,  # proto
                random.choice(['FIN', 'CON', 'INT', 'REQ', 'RST']),  # state
                random.random() * 10,  # dur
                random.randint(100, 10000),  # sbytes
                random.randint(100, 10000),  # dbytes
            ]
            
            # Add remaining columns with random values
            row.extend([random.random() * 100 for _ in range(len(columns) - 9 - 2)])
            
            # Add attack category and label
            attack_cat = random.choice(attack_types)
            row.append(attack_cat)  # attack_cat
            row.append(1)           # label
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Save sample data
        os.makedirs(self.dataset_path, exist_ok=True)
        sample_path = os.path.join(self.dataset_path, "sample_data.csv")
        df.to_csv(sample_path, index=False)
        logger.info(f"Created sample dataset with {len(df)} records at {sample_path}")
        
        return df
    
    def preprocess_data(self, raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Preprocess raw data into a standardized format.
        
        Args:
            raw_data: Raw data from load_raw_data
            
        Returns:
            List of preprocessed data records
        """
        preprocessed_data = []
        
        # Rename columns to standardized format
        column_mapping = {
            'srcip': 'src_ip',
            'sport': 'src_port',
            'dstip': 'dst_ip',
            'dsport': 'dst_port',
            'proto': 'protocol',
            'dur': 'duration',
            'sbytes': 'bytes_sent',
            'dbytes': 'bytes_received',
            'spkts': 'packets_sent',
            'dpkts': 'packets_received',
            'service': 'service',
            'attack_cat': 'attack_category',
            'label': 'is_attack'
        }
        
        # Check if columns exist in the DataFrame
        for old_col, new_col in column_mapping.items():
            if old_col not in raw_data.columns:
                logger.warning(f"Column {old_col} not found in dataset")
        
        # Process each row
        for _, row in raw_data.iterrows():
            record = {}
            
            # Add mapped columns
            for old_col, new_col in column_mapping.items():
                if old_col in row.index:
                    record[new_col] = row[old_col]
            
            # Generate flow ID if not present
            record['flow_id'] = f"{record.get('src_ip', '')}-{record.get('dst_ip', '')}-{record.get('src_port', '')}-{record.get('dst_port', '')}-{record.get('protocol', '')}"
            
            # Add timestamp (not present in UNSW-NB15)
            record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add threat indicators
            record['threat_indicators'] = []
            
            # Check if this is an attack
            if 'attack_category' in record and record['attack_category']:
                record['threat_indicators'].append(f"Attack type: {record['attack_category']}")
            
            # Add to preprocessed data
            preprocessed_data.append(record)
        
        logger.info(f"Preprocessed {len(preprocessed_data)} records")
        return preprocessed_data


class CustomPCAPProcessor(DatasetProcessor):
    """
    Processor for custom PCAP files.
    """
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """
        Initialize the custom PCAP processor.
        
        Args:
            dataset_path: Path to the PCAP files
            output_path: Path to save processed data
        """
        super().__init__("custom_pcap", dataset_path, output_path)
    
    def process(self) -> Dict[str, List[Document]]:
        """
        Process the custom PCAP files.
        
        Returns:
            Dictionary mapping split names to lists of documents
        """
        # Check if processed data already exists
        if os.path.exists(os.path.join(self.output_path, "train.json")):
            logger.info("Processed data already exists, loading from disk")
            return self.load_splits()
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(raw_data)
        
        # Convert to documents
        documents = self.convert_to_documents(preprocessed_data)
        
        # Split data
        splits = self.split_data(documents)
        
        # Save splits
        self.save_splits(splits)
        
        return splits
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw data from custom PCAP files.
        
        Returns:
            List of packet dictionaries
        """
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # List PCAP files in the dataset path
        pcap_files = [
            f for f in os.listdir(self.dataset_path) 
            if f.endswith('.pcap') or f.endswith('.pcapng')
        ]
        
        if not pcap_files:
            logger.warning(f"No PCAP files found in {self.dataset_path}")
            logger.warning("Creating a sample dataset for demonstration purposes")
            return self._create_sample_data()
        
        # In a real implementation, we would use a library like pyshark or scapy
        # to parse PCAP files. For this demonstration, we'll create sample data.
        logger.warning("PCAP parsing not implemented, creating sample data")
        return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """
        Create a sample dataset for demonstration purposes.
        
        Returns:
            List of packet dictionaries
        """
        # Generate sample data
        n_samples = 1000
        data = []
        
        # Generate benign traffic
        for i in range(int(n_samples * 0.7)):
            src_ip = f"192.168.1.{random.randint(1, 254)}"
            dst_ip = f"10.0.0.{random.randint(1, 254)}"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 53, 25])
            protocol = random.choice(['TCP', 'UDP', 'ICMP'])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            packet = {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'timestamp': timestamp,
                'length': random.randint(64, 1500),
                'ttl': random.randint(32, 128),
                'flags': random.choice(['', 'SYN', 'ACK', 'SYN-ACK', 'FIN', 'RST']),
                'is_attack': False
            }
            
            # Add protocol-specific fields
            if protocol == 'TCP':
                packet['tcp_window'] = random.randint(1024, 65535)
                packet['tcp_seq'] = random.randint(1000000, 9999999)
                packet['tcp_ack'] = random.randint(1000000, 9999999)
            elif protocol == 'UDP':
                packet['udp_length'] = random.randint(8, 1472)
            elif protocol == 'ICMP':
                packet['icmp_type'] = random.randint(0, 8)
                packet['icmp_code'] = random.randint(0, 15)
            
            data.append(packet)
        
        # Generate attack traffic
        attack_types = ['Port Scan', 'DoS', 'Brute Force', 'SQL Injection', 'XSS']
        
        for i in range(int(n_samples * 0.3)):
            src_ip = f"192.168.1.{random.randint(1, 254)}"
            dst_ip = f"10.0.0.{random.randint(1, 254)}"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 22, 53, 25])
            protocol = random.choice(['TCP', 'UDP', 'ICMP'])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attack_type = random.choice(attack_types)
            
            packet = {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'timestamp': timestamp,
                'length': random.randint(64, 1500),
                'ttl': random.randint(32, 128),
                'flags': random.choice(['', 'SYN', 'ACK', 'SYN-ACK', 'FIN', 'RST']),
                'is_attack': True,
                'attack_type': attack_type
            }
            
            # Add protocol-specific fields
            if protocol == 'TCP':
                packet['tcp_window'] = random.randint(1024, 65535)
                packet['tcp_seq'] = random.randint(1000000, 9999999)
                packet['tcp_ack'] = random.randint(1000000, 9999999)
            elif protocol == 'UDP':
                packet['udp_length'] = random.randint(8, 1472)
            elif protocol == 'ICMP':
                packet['icmp_type'] = random.randint(0, 8)
                packet['icmp_code'] = random.randint(0, 15)
            
            # Add attack-specific fields
            if attack_type == 'Port Scan':
                packet['scan_type'] = random.choice(['SYN', 'FIN', 'XMAS', 'NULL'])
            elif attack_type == 'DoS':
                packet['dos_method'] = random.choice(['SYN Flood', 'HTTP Flood', 'UDP Flood'])
            elif attack_type == 'Brute Force':
                packet['service'] = random.choice(['SSH', 'FTP', 'HTTP'])
                packet['attempts'] = random.randint(1, 100)
            elif attack_type == 'SQL Injection':
                packet['payload'] = "' OR 1=1 --"
            elif attack_type == 'XSS':
                packet['payload'] = "<script>alert('XSS')</script>"
            
            data.append(packet)
        
        # Save sample data
        os.makedirs(self.dataset_path, exist_ok=True)
        sample_path = os.path.join(self.dataset_path, "sample_data.json")
        with open(sample_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created sample dataset with {len(data)} packets at {sample_path}")
        
        return data
    
    def preprocess_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw data into a standardized format.
        
        Args:
            raw_data: Raw data from load_raw_data
            
        Returns:
            List of preprocessed data records
        """
        preprocessed_data = []
        
        # Group packets by flow
        flows = {}
        
        for packet in raw_data:
            # Create flow key
            flow_key = f"{packet.get('src_ip', '')}-{packet.get('dst_ip', '')}-{packet.get('src_port', '')}-{packet.get('dst_port', '')}-{packet.get('protocol', '')}"
            
            if flow_key not in flows:
                flows[flow_key] = []
            
            flows[flow_key].append(packet)
        
        # Process each flow
        for flow_key, packets in flows.items():
            # Sort packets by timestamp
            packets.sort(key=lambda p: p.get('timestamp', ''))
            
            # Extract flow information
            if not packets:
                continue
            
            first_packet = packets[0]
            last_packet = packets[-1]
            
            # Calculate flow duration
            try:
                first_time = datetime.strptime(first_packet.get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
                last_time = datetime.strptime(last_packet.get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
                duration = (last_time - first_time).total_seconds()
            except:
                duration = 0
            
            # Create flow record
            flow = {
                'flow_id': flow_key,
                'src_ip': first_packet.get('src_ip', ''),
                'dst_ip': first_packet.get('dst_ip', ''),
                'src_port': first_packet.get('src_port', ''),
                'dst_port': first_packet.get('dst_port', ''),
                'protocol': first_packet.get('protocol', ''),
                'timestamp': first_packet.get('timestamp', ''),
                'duration': duration,
                'packets_sent': len(packets),
                'packets_received': 0,  # Simplified
                'bytes_sent': sum(p.get('length', 0) for p in packets),
                'bytes_received': 0,  # Simplified
            }
            
            # Check if this is an attack
            is_attack = any(p.get('is_attack', False) for p in packets)
            flow['is_attack'] = is_attack
            
            if is_attack:
                # Get attack types
                attack_types = set()
                for p in packets:
                    if p.get('is_attack', False) and 'attack_type' in p:
                        attack_types.add(p['attack_type'])
                
                flow['attack_types'] = list(attack_types)
                
                # Add threat indicators
                flow['threat_indicators'] = [f"Attack type: {attack_type}" for attack_type in attack_types]
                
                # Add attack-specific information
                for p in packets:
                    if p.get('is_attack', False):
                        if 'scan_type' in p:
                            flow['scan_type'] = p['scan_type']
                        if 'dos_method' in p:
                            flow['dos_method'] = p['dos_method']
                        if 'service' in p:
                            flow['service'] = p['service']
                        if 'payload' in p:
                            flow['payload'] = p['payload']
            else:
                flow['threat_indicators'] = []
            
            # Add to preprocessed data
            preprocessed_data.append(flow)
        
        logger.info(f"Preprocessed {len(preprocessed_data)} flows from {len(raw_data)} packets")
        return preprocessed_data
