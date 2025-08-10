# Docker Configuration

## Overview
This directory contains Docker configuration files for containerizing the network security RAG comparison research project. These files enable reproducible execution of the code and experiments in a consistent environment.

## Contents
- `Dockerfile` - Main container definition
- `docker-compose.yml` - Multi-container setup for distributed experiments
- `requirements.txt` - Python package dependencies
- `environment.yml` - Conda environment specification
- `scripts/` - Helper scripts for Docker operations
- `.dockerignore` - Files excluded from Docker context

## Container Features
The Docker configuration provides:
- All required dependencies pre-installed
- GPU support for accelerated model training and inference
- Volume mounting for persistent data storage
- Networking for distributed experiments
- Jupyter notebook server for interactive analysis
- Reproducible environment across different host systems

## Usage Instructions

### Basic Usage
```bash
# Build the Docker image
docker build -t network-rag-research .

# Run a container with the current directory mounted
docker run -it --gpus all -v $(pwd):/network-rag-research network-rag-research
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# Run experiments in the container
docker-compose exec research python /network-rag-research/src/evaluation/benchmark.py

# Start Jupyter notebook server
docker-compose exec research jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

## Environment Variables
The following environment variables can be configured:
- `CUDA_VISIBLE_DEVICES` - Control GPU allocation
- `DATA_DIR` - Override default data directory
- `EXPERIMENT_CONFIG` - Specify experiment configuration file
- `LOG_LEVEL` - Set logging verbosity

## Resource Requirements
- Minimum: 8GB RAM, 4 CPU cores, 20GB disk space
- Recommended: 32GB RAM, 8 CPU cores, 100GB disk space, NVIDIA GPU with 8GB+ VRAM

## Troubleshooting
- Check GPU availability with `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`
- Ensure Docker has sufficient resource allocation in Docker Desktop settings
- For permission issues, check volume mount ownership and permissions
- If container exits immediately, check logs with `docker logs <container_id>`

## Customization
The Dockerfile can be customized for different environments:
- Edit `requirements.txt` to modify Python dependencies
- Adjust base image in Dockerfile for different CUDA versions
- Modify resource allocations in docker-compose.yml
