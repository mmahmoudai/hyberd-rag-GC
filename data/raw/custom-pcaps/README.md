# Custom PCAP Datasets

## Overview
This directory contains custom PCAP (Packet Capture) files that were specifically created for this research project to test the RAG architectures in scenarios not covered by public datasets.

## Dataset Information
- **Source**: Custom generated in controlled lab environment
- **Year**: 2025
- **Size**: Varies
- **Format**: PCAP files

## Scenario Types
- Zero-day attack simulations
- Advanced persistent threats
- Multi-stage attacks
- Evasion techniques
- High-throughput network conditions
- Mixed benign and malicious traffic

## Generation Process
These custom PCAPs were generated using:
1. Controlled lab environment with isolated network segments
2. Legitimate user behavior simulation
3. Attack tools and frameworks for malicious traffic
4. Traffic generation tools for volume testing

## Usage in This Project
In this project, we use the custom PCAP datasets to:
1. Test RAG models against scenarios not present in public datasets
2. Evaluate performance under adversarial conditions
3. Assess detection capabilities for novel attack patterns
4. Benchmark high-throughput performance

## Preprocessing Steps
The raw PCAP files should be processed using the scripts in `/network-rag-research/src/data_processing/` to:
1. Extract relevant features
2. Normalize data
3. Split into training and testing sets
4. Convert to appropriate format for RAG models

## Confidentiality Notice
These custom datasets may contain sensitive information about attack techniques and network vulnerabilities. They should be handled according to the project's data security guidelines and not shared publicly without proper anonymization.
