# UNSW-NB15 Dataset

## Overview
The UNSW-NB15 dataset is a network traffic dataset created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS). It contains a mix of real modern normal activities and synthetic contemporary attack behaviors.

## Dataset Information
- **Source**: Australian Centre for Cyber Security (ACCS)
- **Year**: 2015
- **Size**: Approximately 100GB
- **Format**: PCAP files and CSV files with extracted features

## Attack Types
- Fuzzers
- Analysis
- Backdoors
- DoS
- Exploits
- Generic
- Reconnaissance
- Shellcode
- Worms

## Download Instructions
The dataset can be downloaded from the official UNSW website:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Usage in This Project
In this project, we use the UNSW-NB15 dataset to:
1. Train and evaluate RAG models for network security analysis
2. Test detection capabilities for various attack types
3. Benchmark performance across different RAG architectures
4. Validate findings from the CIC-IDS2017 dataset

## Preprocessing Steps
The raw PCAP files should be processed using the scripts in `/network-rag-research/src/data_processing/` to:
1. Extract relevant features
2. Normalize data
3. Split into training and testing sets
4. Convert to appropriate format for RAG models

## Citation
```
Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.
```
