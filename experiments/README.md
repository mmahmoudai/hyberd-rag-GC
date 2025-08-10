# Experiments Directory

## Overview
This directory contains configuration files, logs, and results for all experiments conducted in the network security RAG comparison research project. It provides a structured way to organize experimental data and ensure reproducibility.

## Directory Structure
- `/configs/` - Configuration files for different experiments
- `/logs/` - Training and evaluation logs
- `/results/` - Metrics and performance data

## Configuration Files
The `/configs/` directory contains JSON or YAML files that define:
- Model parameters for each RAG variant
- Dataset selection and preprocessing options
- Evaluation metrics and thresholds
- Test scenario parameters
- Ablation study configurations

Each configuration file is versioned and includes detailed comments to explain parameter choices.

## Log Files
The `/logs/` directory contains:
- Training logs with loss values and optimization details
- Evaluation logs with intermediate results
- Error logs for debugging
- System performance logs (CPU, memory, disk usage)
- Timing information for benchmarking

Log files follow a consistent naming convention: `{experiment_type}_{rag_variant}_{timestamp}.log`

## Results
The `/results/` directory contains:
- CSV files with performance metrics
- JSON files with detailed evaluation results
- Serialized model outputs for analysis
- Comparison tables across RAG variants
- Statistical significance test results
- Ablation study outcomes

Results are organized by experiment type and RAG variant for easy comparison.

## Reproducibility
To reproduce any experiment:
1. Select the appropriate configuration file from `/configs/`
2. Run the experiment using scripts in `/network-rag-research/src/evaluation/`
3. Compare your results with the stored results in `/results/`

## Best Practices
- Never modify original result files; create new ones for each experiment run
- Include configuration file references in result filenames
- Document any deviations from the standard experimental setup
- Use version control to track changes to configuration files
- Archive important results with detailed notes on experimental conditions
