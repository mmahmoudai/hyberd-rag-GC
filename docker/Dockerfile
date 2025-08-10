FROM python:3.8-slim

WORKDIR /network-rag-research

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p src/models/traditional_rag \
    src/models/graph_rag \
    src/models/cache_rag \
    src/models/hybrid_rag \
    src/models/common \
    src/data_processing \
    src/evaluation \
    src/visualization \
    src/utils \
    src/experiments \
    data/raw/cic-ids2017 \
    data/raw/unsw-nb15 \
    data/raw/custom-pcaps \
    data/processed \
    data/embeddings \
    experiments/configs \
    experiments/logs \
    experiments/results \
    experiments/ablation \
    notebooks \
    papers/ieee \
    papers/springer \
    papers/figures

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/network-rag-research"

# Default command
CMD ["bash"]
