# Processed Data Directory

## Overview
This directory contains preprocessed versions of the network security datasets used in this research project. The processed data is ready for direct use in training and evaluating RAG models.

## Directory Structure
- `/cic-ids2017/` - Processed CIC-IDS2017 dataset
- `/unsw-nb15/` - Processed UNSW-NB15 dataset
- `/custom-pcaps/` - Processed custom PCAP datasets

## Processing Pipeline
All raw datasets have been processed through a standardized pipeline that includes:

1. **Feature Extraction**: Converting raw packet data into structured features
2. **Normalization**: Scaling numerical features to appropriate ranges
3. **Labeling**: Ensuring consistent attack type labeling across datasets
4. **Chunking**: Dividing data into appropriate document chunks for RAG systems
5. **Filtering**: Removing irrelevant or redundant information
6. **Format Conversion**: Converting to formats optimized for RAG processing

## Usage Guidelines
- Use the processed data directly with the RAG models in `/network-rag-research/src/models/`
- Each subdirectory contains dataset-specific README files with detailed information
- Scripts for reproducing the processing pipeline are available in `/network-rag-research/src/data_processing/`

## Data Format
The processed data is stored in the following formats:
- JSON files for document collections
- CSV files for structured feature data
- Pickle files for serialized Python objects
- HDF5 files for large numerical arrays

## Versioning
Each processed dataset includes version information to ensure reproducibility. When using processed data in experiments, always record the version number in your configuration files.
