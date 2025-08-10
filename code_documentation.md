# Network Security RAG Comparison - Code Documentation

**Contact:** Dr Muhammad M.Hanafy  
**Email:** m.mahmoud@mau.edu.eg

This document provides comprehensive documentation for the code implementation of our network security RAG comparison research project. The documentation covers all components of the system, including models, data processing, evaluation, and visualization.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Directory Structure](#directory-structure)
4. [Models](#models)
5. [Data Processing](#data-processing)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)
8. [Experiments](#experiments)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)

## Project Overview

This project implements and compares four Retrieval Augmented Generation (RAG) architectures for network security packet analysis:

1. **Traditional RAG**: A vector-based retrieval system optimized for network security data
2. **Graph RAG**: A graph-based approach that models network entities and their relationships
3. **Cache RAG**: A performance-optimized system with strategic caching mechanisms
4. **Hybrid Cache-Graph RAG**: A novel approach combining graph-based knowledge representation with caching benefits

The project includes implementations of all four architectures, data processing pipelines, evaluation frameworks, and visualization tools.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 16GB+ RAM
- 100GB+ disk space

### Setup with Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/network-rag-research/network-security-rag-comparison.git
cd network-rag-research

# Build and run the Docker container
docker build -t network-rag-research .
docker run -it --gpus all -v $(pwd):/network-rag-research network-rag-research
```

### Manual Setup
```bash
# Clone the repository
git clone https://github.com/network-rag-research/network-security-rag-comparison.git
cd network-rag-research

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
/network-rag-research/
├── src/
│   ├── models/              # Implementation of all RAG variants
│   ├── data_processing/     # Data preprocessing and loading utilities
│   ├── evaluation/          # Benchmarking and testing code
│   ├── visualization/       # Plotting and graph generation code
│   └── utils/               # Helper functions and common utilities
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned and preprocessed data
│   └── embeddings/          # Vector representations and indexes
├── experiments/
│   ├── configs/             # Configuration files for different experiments
│   ├── logs/                # Training and evaluation logs
│   └── results/             # Metrics and performance data
└── docker/                  # Containerization files for reproducibility
```

## Models

### Traditional RAG

The Traditional RAG architecture follows the standard retrieval-augmented generation paradigm, adapted for network security packet analysis.

#### Key Components

- **Document Processing**: Converts network packets and flow records into document chunks
- **Embedding Generation**: Encodes documents into dense vector representations
- **Vector Index**: Stores embeddings for efficient similarity search
- **Generation**: Uses retrieved documents as context for a language model

#### Usage Example

```python
from src.models.traditional_rag import TraditionalRAG

# Initialize the model
trad_rag = TraditionalRAG(
    embedding_dim=768,
    chunk_strategy="sliding_window",
    retrieval_method="dense"
)

# Process documents
trad_rag.process_documents(documents)

# Query the system
results = trad_rag.query("Detect potential DDoS attacks in this traffic")
print(results)
```

### Graph RAG

The Graph RAG architecture extends Traditional RAG by modeling network entities and their relationships as a knowledge graph.

#### Key Components

- **Knowledge Graph Construction**: Represents network entities as nodes in a heterogeneous graph
- **Graph Embedding**: Generates node and edge embeddings using a graph neural network
- **Graph Traversal Retrieval**: Performs multi-hop traversal to retrieve related entities
- **Context Aggregation**: Aggregates information from retrieved nodes and their neighborhoods

#### Usage Example

```python
from src.models.graph_rag import GraphRAG

# Initialize the model
graph_rag = GraphRAG(
    graph_type="heterogeneous",
    node_config="entity_attribute",
    traversal_depth=2
)

# Build the knowledge graph
graph_rag.build_graph(network_data)

# Query the system
results = graph_rag.query("Identify communication patterns between these IP addresses")
print(results)
```

### Cache RAG

The Cache RAG architecture focuses on performance optimization through strategic caching mechanisms.

#### Key Components

- **Query Cache**: Stores frequently asked queries and their results
- **Embedding Cache**: Caches document embeddings to avoid recomputation
- **Result Cache**: Stores generation results for specific query-document combinations
- **Cache Invalidation Strategy**: Ensures cache coherence while maximizing performance

#### Usage Example

```python
from src.models.cache_rag import CacheRAG

# Initialize the model
cache_rag = CacheRAG(
    cache_size=10000,
    ttl_minutes=10,
    invalidation_policy="hybrid"
)

# Process documents
cache_rag.process_documents(documents)

# Query the system
results = cache_rag.query("Summarize recent port scan activities")
print(results)
```

### Hybrid Cache-Graph RAG

Our novel Hybrid Cache-Graph RAG architecture combines the contextual understanding of Graph RAG with the performance benefits of Cache RAG.

#### Key Components

- **Integrated Knowledge Representation**: Maintains both vector embeddings and a knowledge graph
- **Adaptive Retrieval Strategy**: Dynamically selects the optimal retrieval strategy
- **Graph-Aware Caching**: Enhances caching with graph awareness
- **Adaptive Component Weighting**: Dynamically adjusts component weights based on historical performance

#### Usage Example

```python
from src.models.hybrid_rag import HybridCacheGraphRAG

# Initialize the model
hybrid_rag = HybridCacheGraphRAG(
    component_weights="adaptive",
    integration_strategy="hierarchical",
    adaptation_rate="medium"
)

# Process documents and build graph
hybrid_rag.process_data(network_data)

# Query the system
results = hybrid_rag.query("Analyze potential data exfiltration in this traffic pattern")
print(results)
```

## Data Processing

The data processing module handles the preparation of network security datasets for use with the RAG architectures.

### Dataset Processors

- **CIC-IDS2017 Processor**: Processes the CIC-IDS2017 dataset
- **UNSW-NB15 Processor**: Processes the UNSW-NB15 dataset
- **PCAP Processor**: Processes custom PCAP files

### Usage Example

```python
from src.data_processing.dataset_processing import CICProcessor, UNSWProcessor, PCAPProcessor

# Process CIC-IDS2017 dataset
cic_processor = CICProcessor(raw_path="data/raw/cic-ids2017", processed_path="data/processed/cic-ids2017")
cic_data = cic_processor.process()

# Process UNSW-NB15 dataset
unsw_processor = UNSWProcessor(raw_path="data/raw/unsw-nb15", processed_path="data/processed/unsw-nb15")
unsw_data = unsw_processor.process()

# Process custom PCAP files
pcap_processor = PCAPProcessor(raw_path="data/raw/custom-pcaps", processed_path="data/processed/custom-pcaps")
pcap_data = pcap_processor.process()
```

### Feature Extraction

The feature extraction module extracts relevant features from network packets and flows.

```python
from src.data_processing.feature_extraction import PacketFeatureExtractor

# Initialize feature extractor
extractor = PacketFeatureExtractor()

# Extract features from packets
features = extractor.extract_features(packets)
```

### Data Splitting

The data splitting module handles the division of datasets into training and testing sets.

```python
from src.data_processing.data_splitting import DataSplitter

# Initialize data splitter
splitter = DataSplitter(test_size=0.2, random_state=42)

# Split data into training and testing sets
train_data, test_data = splitter.split(processed_data)
```

## Evaluation

The evaluation module provides tools for benchmarking and testing the RAG architectures.

### Metrics

- **Retrieval Metrics**: MAP, MRR, Precision@k, Recall@k
- **Classification Metrics**: Precision, Recall, F1-score, AUC
- **Efficiency Metrics**: Latency, Throughput, Memory usage, CPU utilization

### Test Scenarios

- **Standard Scenario**: Evaluates performance on known attack types
- **Zero-Day Scenario**: Tests ability to detect novel threats
- **High-Throughput Scenario**: Evaluates scalability under high traffic volumes
- **Adversarial Scenario**: Tests resilience against evasion techniques

### Usage Example

```python
from src.evaluation.benchmark import Benchmark
from src.evaluation.metrics import RetrievalMetrics, ClassificationMetrics, EfficiencyMetrics

# Initialize benchmark
benchmark = Benchmark(
    models=[trad_rag, graph_rag, cache_rag, hybrid_rag],
    metrics=[RetrievalMetrics(), ClassificationMetrics(), EfficiencyMetrics()],
    scenarios=["standard", "zero_day", "high_throughput", "adversarial"]
)

# Run benchmark
results = benchmark.run(test_data)

# Print results
benchmark.print_results(results)
```

### Ablation Studies

The ablation module allows for isolating the contribution of individual components in each architecture.

```python
from src.evaluation.ablation import AblationStudy

# Initialize ablation study
ablation = AblationStudy(model=hybrid_rag)

# Define components to ablate
components = {
    "component_weights": ["fixed", "adaptive"],
    "integration_strategy": ["parallel", "sequential", "hierarchical"],
    "adaptation_rate": ["fast", "medium", "slow"]
}

# Run ablation study
ablation_results = ablation.run(components, test_data)

# Print results
ablation.print_results(ablation_results)
```

## Visualization

The visualization module provides tools for generating visualizations of the RAG architectures and their performance.

### System Architecture Diagrams

```python
from src.visualization.system_architecture import ArchitectureDiagramGenerator

# Initialize diagram generator
diagram_gen = ArchitectureDiagramGenerator()

# Generate diagrams
diagram_gen.generate_traditional_rag_diagram(output_path="papers/figures/system_architecture/traditional_rag.svg")
diagram_gen.generate_graph_rag_diagram(output_path="papers/figures/system_architecture/graph_rag.svg")
diagram_gen.generate_cache_rag_diagram(output_path="papers/figures/system_architecture/cache_rag.svg")
diagram_gen.generate_hybrid_rag_diagram(output_path="papers/figures/system_architecture/hybrid_cache_graph_rag.svg")
```

### Performance Charts

```python
from src.visualization.performance_charts import PerformanceChartGenerator

# Initialize chart generator
chart_gen = PerformanceChartGenerator()

# Generate performance charts
chart_gen.generate_accuracy_comparison(results, output_path="papers/figures/performance_comparison/accuracy_comparison.svg")
chart_gen.generate_roc_curves(results, output_path="papers/figures/performance_comparison/roc_curves.svg")
chart_gen.generate_pr_curves(results, output_path="papers/figures/performance_comparison/pr_curves.svg")
chart_gen.generate_latency_comparison(results, output_path="papers/figures/performance_comparison/latency_comparison.svg")
```

### Knowledge Graph Visualizations

```python
from src.visualization.knowledge_graphs import KnowledgeGraphVisualizer

# Initialize visualizer
kg_vis = KnowledgeGraphVisualizer()

# Generate knowledge graph visualizations
kg_vis.visualize_entity_relationships(graph, output_path="papers/figures/knowledge_graphs/entity_relationships.svg")
kg_vis.visualize_communication_patterns(graph, output_path="papers/figures/knowledge_graphs/communication_patterns.svg")
kg_vis.visualize_subgraph_extraction(graph, query, output_path="papers/figures/knowledge_graphs/subgraph_extraction.svg")
```

### Cache Performance Visualizations

```python
from src.visualization.cache_performance import CachePerformanceVisualizer

# Initialize visualizer
cache_vis = CachePerformanceVisualizer()

# Generate cache performance visualizations
cache_vis.visualize_hit_rate_heatmap(cache_results, output_path="papers/figures/cache_performance/hit_rate_size_complexity.svg")
cache_vis.visualize_invalidation_timeline(cache_results, output_path="papers/figures/cache_performance/invalidation_timeline.svg")
cache_vis.visualize_time_series(cache_results, output_path="papers/figures/cache_performance/hit_rate_time_series.svg")
```

### Graph Traversal Analytics

```python
from src.visualization.graph_traversal import GraphTraversalVisualizer

# Initialize visualizer
graph_vis = GraphTraversalVisualizer()

# Generate graph traversal visualizations
graph_vis.visualize_hop_distance_histogram(graph_results, output_path="papers/figures/graph_traversal/hop_distance_histogram.svg")
graph_vis.visualize_hop_count_accuracy(graph_results, output_path="papers/figures/graph_traversal/hop_count_accuracy.svg")
graph_vis.visualize_topk_optimization(graph_results, output_path="papers/figures/graph_traversal/topk_optimization.svg")
```

### Experimental Results Visualizations

```python
from src.visualization.experimental_results import ExperimentalResultsVisualizer

# Initialize visualizer
exp_vis = ExperimentalResultsVisualizer()

# Generate experimental results visualizations
exp_vis.visualize_scenario_comparison(results, output_path="papers/figures/experimental_results/scenario_comparison.svg")
exp_vis.visualize_ablation_study(ablation_results, output_path="papers/figures/experimental_results/ablation_study_hybrid_cache-graph_rag.svg")
exp_vis.visualize_confusion_matrices(results, output_path="papers/figures/experimental_results/confusion_matrix_hybrid_cache-graph_rag.svg")
exp_vis.visualize_significance_tests(significance_results, output_path="papers/figures/experimental_results/significance_pvalues.svg")
```

## Experiments

The experiments module provides tools for running experiments with the RAG architectures.

### Configuration

Experiment configurations are stored in YAML files in the `experiments/configs` directory.

Example configuration file (`experiments/configs/hybrid_rag_zero_day.yaml`):
```yaml
model:
  type: hybrid_cache_graph_rag
  params:
    component_weights: adaptive
    integration_strategy: hierarchical
    adaptation_rate: medium

dataset:
  name: custom_pcaps
  scenario: zero_day
  split:
    test_size: 0.2
    random_state: 42

evaluation:
  metrics:
    - map
    - mrr
    - precision@10
    - recall@10
    - f1_score
    - auc
    - latency
    - throughput
  n_runs: 5
  significance_test: true
```

### Running Experiments

```python
from src.experiments.run_experiments import ExperimentRunner

# Initialize experiment runner
runner = ExperimentRunner()

# Run experiment from configuration file
results = runner.run_from_config("experiments/configs/hybrid_rag_zero_day.yaml")

# Save results
runner.save_results(results, "experiments/results/hybrid_rag_zero_day_results.json")
```

### Analyzing Results

```python
from src.experiments.analyze_results import ResultAnalyzer

# Initialize result analyzer
analyzer = ResultAnalyzer()

# Load results
results = analyzer.load_results("experiments/results/hybrid_rag_zero_day_results.json")

# Analyze results
analysis = analyzer.analyze(results)

# Generate report
analyzer.generate_report(analysis, "experiments/results/hybrid_rag_zero_day_report.md")
```

## API Reference

### Models

#### TraditionalRAG

```python
class TraditionalRAG:
    def __init__(self, embedding_dim=768, chunk_strategy="sliding_window", retrieval_method="dense"):
        """
        Initialize the Traditional RAG model.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            chunk_strategy (str): Strategy for chunking documents ("fixed_size", "sliding_window", "semantic")
            retrieval_method (str): Method for retrieval ("bm25", "dense", "hybrid")
        """
        
    def process_documents(self, documents):
        """
        Process documents for retrieval.
        
        Args:
            documents (list): List of documents to process
        """
        
    def query(self, query_text):
        """
        Query the system.
        
        Args:
            query_text (str): Query text
            
        Returns:
            dict: Query results
        """
```

#### GraphRAG

```python
class GraphRAG:
    def __init__(self, graph_type="heterogeneous", node_config="entity_attribute", traversal_depth=2):
        """
        Initialize the Graph RAG model.
        
        Args:
            graph_type (str): Type of graph ("homogeneous", "heterogeneous")
            node_config (str): Node configuration ("entity_only", "entity_attribute")
            traversal_depth (int): Depth of graph traversal
        """
        
    def build_graph(self, network_data):
        """
        Build the knowledge graph.
        
        Args:
            network_data (dict): Network data to build the graph from
        """
        
    def query(self, query_text):
        """
        Query the system.
        
        Args:
            query_text (str): Query text
            
        Returns:
            dict: Query results
        """
```

#### CacheRAG

```python
class CacheRAG:
    def __init__(self, cache_size=10000, ttl_minutes=10, invalidation_policy="hybrid"):
        """
        Initialize the Cache RAG model.
        
        Args:
            cache_size (int): Size of the cache in number of entries
            ttl_minutes (int): Time-to-live in minutes
            invalidation_policy (str): Cache invalidation policy ("time_based", "change_based", "hybrid")
        """
        
    def process_documents(self, documents):
        """
        Process documents for retrieval.
        
        Args:
            documents (list): List of documents to process
        """
        
    def query(self, query_text):
        """
        Query the system.
        
        Args:
            query_text (str): Query text
            
        Returns:
            dict: Query results
        """
```

#### HybridCacheGraphRAG

```python
class HybridCacheGraphRAG:
    def __init__(self, component_weights="adaptive", integration_strategy="hierarchical", adaptation_rate="medium"):
        """
        Initialize the Hybrid Cache-Graph RAG model.
        
        Args:
            component_weights (str): Component weighting strategy ("fixed", "adaptive")
            integration_strategy (str): Integration strategy ("parallel", "sequential", "hierarchical")
            adaptation_rate (str): Rate of adaptation ("fast", "medium", "slow")
        """
        
    def process_data(self, network_data):
        """
        Process data for retrieval and build the graph.
        
        Args:
            network_data (dict): Network data to process
        """
        
    def query(self, query_text):
        """
        Query the system.
        
        Args:
            query_text (str): Query text
            
        Returns:
            dict: Query results
        """
```

### Data Processing

#### DatasetProcessor

```python
class DatasetProcessor:
    def __init__(self, raw_path, processed_path):
        """
        Initialize the dataset processor.
        
        Args:
            raw_path (str): Path to raw dataset
            processed_path (str): Path to save processed dataset
        """
        
    def process(self):
        """
        Process the dataset.
        
        Returns:
            dict: Processed data
        """
```

#### FeatureExtractor

```python
class FeatureExtractor:
    def __init__(self):
        """
        Initialize the feature extractor.
        """
        
    def extract_features(self, data):
        """
        Extract features from data.
        
        Args:
            data (dict): Data to extract features from
            
        Returns:
            dict: Extracted features
        """
```

#### DataSplitter

```python
class DataSplitter:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the data splitter.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        
    def split(self, data):
        """
        Split data into training and testing sets.
        
        Args:
            data (dict): Data to split
            
        Returns:
            tuple: (train_data, test_data)
        """
```

### Evaluation

#### Benchmark

```python
class Benchmark:
    def __init__(self, models, metrics, scenarios):
        """
        Initialize the benchmark.
        
        Args:
            models (list): List of models to benchmark
            metrics (list): List of metrics to compute
            scenarios (list): List of test scenarios
        """
        
    def run(self, test_data):
        """
        Run the benchmark.
        
        Args:
            test_data (dict): Test data
            
        Returns:
            dict: Benchmark results
        """
        
    def print_results(self, results):
        """
        Print benchmark results.
        
        Args:
            results (dict): Benchmark results
        """
```

#### AblationStudy

```python
class AblationStudy:
    def __init__(self, model):
        """
        Initialize the ablation study.
        
        Args:
            model: Model to study
        """
        
    def run(self, components, test_data):
        """
        Run the ablation study.
        
        Args:
            components (dict): Components to ablate
            test_data (dict): Test data
            
        Returns:
            dict: Ablation study results
        """
        
    def print_results(self, results):
        """
        Print ablation study results.
        
        Args:
            results (dict): Ablation study results
        """
```

### Visualization

#### VisualizationBase

```python
class VisualizationBase:
    def __init__(self):
        """
        Initialize the visualization base.
        """
        
    def save_figure(self, fig, output_path):
        """
        Save figure to file.
        
        Args:
            fig: Figure to save
            output_path (str): Path to save figure
        """
```
