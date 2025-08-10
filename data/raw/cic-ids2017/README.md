# CIC-IDS2017 Dataset

## Overview
The CIC-IDS2017 dataset contains network traffic data including benign and various attack scenarios. It was created by the Canadian Institute for Cybersecurity (CIC) and includes a diverse set of attack types.

## Dataset Information
- **Source**: Canadian Institute for Cybersecurity (CIC)
- **Year**: 2017
- **Size**: Approximately 80GB
- **Format**: PCAP files and CSV files with extracted features

## Attack Types
- Brute Force
- Heartbleed
- Botnet
- DoS / DDoS
- Web Attacks
- Infiltration
- Port Scan

## Download Instructions
The dataset can be downloaded from the official CIC website:
https://www.unb.ca/cic/datasets/ids-2017.html

## Usage in This Project
In this project, we use the CIC-IDS2017 dataset to:
1. Train and evaluate RAG models for network security analysis
2. Test detection capabilities for various attack types
3. Benchmark performance across different RAG architectures

## Preprocessing Steps
The raw PCAP files should be processed using the scripts in `/network-rag-research/src/data_processing/` to:
1. Extract relevant features
2. Normalize data
3. Split into training and testing sets
4. Convert to appropriate format for RAG models

## Citation
```
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018
```
