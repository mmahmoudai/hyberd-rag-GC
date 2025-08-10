"""
Evaluation metrics for RAG systems in network security packet analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json

from ..common.base import Document, Query, RetrievalResult, RAGSystem

# Configure logging
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Class for computing evaluation metrics for RAG systems.
    """
    @staticmethod
    def compute_retrieval_metrics(
        retrieved_docs: List[Document],
        relevant_docs: List[Document],
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant documents (ground truth)
            k_values: List of k values for precision@k and recall@k
            
        Returns:
            Dictionary of metrics
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50]
        
        # Get document IDs
        retrieved_ids = [doc.id for doc in retrieved_docs]
        relevant_ids = [doc.id for doc in relevant_docs]
        
        # Compute metrics
        metrics = {}
        
        # Precision and recall at different k values
        for k in k_values:
            if k <= len(retrieved_ids):
                retrieved_at_k = retrieved_ids[:k]
                
                # Precision@k
                precision_at_k = len(set(retrieved_at_k) & set(relevant_ids)) / len(retrieved_at_k)
                metrics[f"precision@{k}"] = precision_at_k
                
                # Recall@k
                if relevant_ids:
                    recall_at_k = len(set(retrieved_at_k) & set(relevant_ids)) / len(relevant_ids)
                    metrics[f"recall@{k}"] = recall_at_k
                else:
                    metrics[f"recall@{k}"] = 0.0
        
        # Mean Average Precision (MAP)
        if relevant_ids:
            avg_precision = 0.0
            num_relevant_retrieved = 0
            
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    num_relevant_retrieved += 1
                    precision_at_i = num_relevant_retrieved / (i + 1)
                    avg_precision += precision_at_i
            
            if num_relevant_retrieved > 0:
                metrics["map"] = avg_precision / len(relevant_ids)
            else:
                metrics["map"] = 0.0
        else:
            metrics["map"] = 0.0
        
        # Mean Reciprocal Rank (MRR)
        if relevant_ids:
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    metrics["mrr"] = 1.0 / (i + 1)
                    break
            else:
                metrics["mrr"] = 0.0
        else:
            metrics["mrr"] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_classification_metrics(
        predictions: List[bool],
        ground_truth: List[bool]
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: List of predicted labels (True for attack, False for benign)
            ground_truth: List of ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy arrays
        y_pred = np.array(predictions)
        y_true = np.array(ground_truth)
        
        # Accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        
        return metrics
    
    @staticmethod
    def compute_performance_metrics(
        latencies: List[float],
        memory_usages: List[float]
    ) -> Dict[str, float]:
        """
        Compute performance metrics.
        
        Args:
            latencies: List of latency measurements (seconds)
            memory_usages: List of memory usage measurements (MB)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Latency statistics
        metrics["mean_latency"] = np.mean(latencies)
        metrics["median_latency"] = np.median(latencies)
        metrics["min_latency"] = np.min(latencies)
        metrics["max_latency"] = np.max(latencies)
        metrics["p95_latency"] = np.percentile(latencies, 95)
        metrics["p99_latency"] = np.percentile(latencies, 99)
        
        # Memory usage statistics
        metrics["mean_memory"] = np.mean(memory_usages)
        metrics["median_memory"] = np.median(memory_usages)
        metrics["min_memory"] = np.min(memory_usages)
        metrics["max_memory"] = np.max(memory_usages)
        metrics["p95_memory"] = np.percentile(memory_usages, 95)
        
        return metrics
    
    @staticmethod
    def compute_cache_metrics(
        cache_hits: int,
        cache_misses: int,
        cache_size: int,
        cache_evictions: int
    ) -> Dict[str, float]:
        """
        Compute cache-specific metrics.
        
        Args:
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            cache_size: Cache size
            cache_evictions: Number of cache evictions
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Hit ratio
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            metrics["hit_ratio"] = cache_hits / total_requests
        else:
            metrics["hit_ratio"] = 0.0
        
        # Miss ratio
        if total_requests > 0:
            metrics["miss_ratio"] = cache_misses / total_requests
        else:
            metrics["miss_ratio"] = 0.0
        
        # Eviction rate
        if total_requests > 0:
            metrics["eviction_rate"] = cache_evictions / total_requests
        else:
            metrics["eviction_rate"] = 0.0
        
        # Cache utilization
        metrics["cache_size"] = cache_size
        
        return metrics
    
    @staticmethod
    def compute_graph_metrics(
        graph_size: int,
        node_types: Dict[str, int],
        edge_types: Dict[str, int],
        query_traversal_depth: float,
        traversal_time: float
    ) -> Dict[str, float]:
        """
        Compute graph-specific metrics.
        
        Args:
            graph_size: Total number of nodes in the graph
            node_types: Dictionary mapping node types to counts
            edge_types: Dictionary mapping edge types to counts
            query_traversal_depth: Average traversal depth for queries
            traversal_time: Average traversal time
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Graph size
        metrics["graph_size"] = graph_size
        
        # Node type distribution
        for node_type, count in node_types.items():
            metrics[f"node_type_{node_type}"] = count
        
        # Edge type distribution
        for edge_type, count in edge_types.items():
            metrics[f"edge_type_{edge_type}"] = count
        
        # Traversal metrics
        metrics["avg_traversal_depth"] = query_traversal_depth
        metrics["avg_traversal_time"] = traversal_time
        
        return metrics
    
    @staticmethod
    def compute_hybrid_metrics(
        cache_weight: float,
        graph_weight: float,
        synergy_factor: float,
        cache_hit_rate: float,
        graph_hit_rate: float,
        hybrid_hit_rate: float
    ) -> Dict[str, float]:
        """
        Compute hybrid-specific metrics.
        
        Args:
            cache_weight: Weight assigned to cache component
            graph_weight: Weight assigned to graph component
            synergy_factor: Synergy factor between cache and graph
            cache_hit_rate: Hit rate for cache-only queries
            graph_hit_rate: Hit rate for graph-only queries
            hybrid_hit_rate: Hit rate for hybrid queries
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Component weights
        metrics["cache_weight"] = cache_weight
        metrics["graph_weight"] = graph_weight
        
        # Synergy factor
        metrics["synergy_factor"] = synergy_factor
        
        # Hit rates
        metrics["cache_hit_rate"] = cache_hit_rate
        metrics["graph_hit_rate"] = graph_hit_rate
        metrics["hybrid_hit_rate"] = hybrid_hit_rate
        
        # Synergy benefit
        expected_hit_rate = (cache_weight * cache_hit_rate) + (graph_weight * graph_hit_rate)
        if expected_hit_rate > 0:
            metrics["synergy_benefit"] = hybrid_hit_rate / expected_hit_rate
        else:
            metrics["synergy_benefit"] = 1.0
        
        return metrics


class PerformanceMonitor:
    """
    Class for monitoring system performance during evaluation.
    """
    def __init__(self):
        """Initialize the performance monitor."""
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None
        self.cpu_usages = []
        self.memory_usages = []
        self.sampling_interval = 0.1  # seconds
        self._monitoring = False
    
    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self.cpu_usages = []
        self.memory_usages = []
        self._monitoring = True
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.end_time = time.time()
        self.end_memory = self._get_memory_usage()
        self._monitoring = False
    
    def sample(self) -> None:
        """Sample current performance metrics."""
        if self._monitoring:
            cpu_percent = psutil.cpu_percent()
            memory_usage = self._get_memory_usage()
            
            self.cpu_usages.append(cpu_percent)
            self.memory_usages.append(memory_usage)
            
            if memory_usage > self.peak_memory:
                self.peak_memory = memory_usage
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        if self.start_time is not None and self.end_time is not None:
            # Execution time
            metrics["execution_time"] = self.end_time - self.start_time
            
            # Memory usage
            metrics["start_memory_mb"] = self.start_memory
            metrics["peak_memory_mb"] = self.peak_memory
            metrics["end_memory_mb"] = self.end_memory
            metrics["memory_increase_mb"] = self.end_memory - self.start_memory
            
            # CPU usage
            if self.cpu_usages:
                metrics["mean_cpu_percent"] = np.mean(self.cpu_usages)
                metrics["max_cpu_percent"] = np.max(self.cpu_usages)
            
            # Memory usage over time
            if self.memory_usages:
                metrics["mean_memory_mb"] = np.mean(self.memory_usages)
                metrics["median_memory_mb"] = np.median(self.memory_usages)
                metrics["p95_memory_mb"] = np.percentile(self.memory_usages, 95)
        
        return metrics


class RAGEvaluator:
    """
    Class for evaluating RAG systems.
    """
    def __init__(
        self,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir or Path("/home/ubuntu/research/experiments/results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.performance_monitor = PerformanceMonitor()
        self.results = {}
    
    def evaluate_system(
        self,
        system: RAGSystem,
        system_name: str,
        queries: List[Query],
        ground_truth: Dict[str, List[Document]],
        test_scenario: str
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG system.
        
        Args:
            system: RAG system to evaluate
            system_name: Name of the system
            queries: List of queries to evaluate
            ground_truth: Dictionary mapping query IDs to relevant documents
            test_scenario: Name of the test scenario
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {system_name} on {test_scenario} scenario with {len(queries)} queries")
        
        results = {
            "system_name": system_name,
            "test_scenario": test_scenario,
            "num_queries": len(queries),
            "retrieval_metrics": {},
            "classification_metrics": {},
            "performance_metrics": {},
            "query_results": []
        }
        
        # Track performance metrics
        latencies = []
        memory_usages = []
        
        # Track classification metrics
        predictions = []
        ground_truth_labels = []
        
        # Process each query
        for i, query in enumerate(queries):
            query_id = getattr(query, 'id', str(i))
            
            # Get ground truth for this query
            relevant_docs = ground_truth.get(query_id, [])
            
            # Start performance monitoring
            self.performance_monitor.start()
            
            # Process query
            start_time = time.time()
            response = system.process_query(query)
            end_time = time.time()
            
            # Stop performance monitoring
            self.performance_monitor.stop()
            
            # Extract retrieved documents
            if 'sources' in response:
                retrieved_docs = [
                    Document(
                        id=source['id'],
                        content='',
                        metadata=source.get('metadata', {})
                    )
                    for source in response['sources']
                ]
            else:
                retrieved_docs = []
            
            # Calculate latency
            latency = end_time - start_time
            latencies.append(latency)
            
            # Get memory usage
            memory_usage = self.performance_monitor.peak_memory
            memory_usages.append(memory_usage)
            
            # Calculate retrieval metrics for this query
            query_retrieval_metrics = EvaluationMetrics.compute_retrieval_metrics(
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs
            )
            
            # Extract classification prediction
            is_attack_prediction = False
            if 'potential_threats' in response and response['potential_threats']:
                is_attack_prediction = True
            
            # Get ground truth label
            is_attack_ground_truth = False
            for doc in relevant_docs:
                if doc.metadata.get('is_attack', False):
                    is_attack_ground_truth = True
                    break
            
            # Add to classification metrics
            predictions.append(is_attack_prediction)
            ground_truth_labels.append(is_attack_ground_truth)
            
            # Store query result
            query_result = {
                "query_id": query_id,
                "query_text": query.text,
                "retrieval_metrics": query_retrieval_metrics,
                "latency": latency,
                "memory_usage": memory_usage,
                "is_attack_prediction": is_attack_prediction,
                "is_attack_ground_truth": is_attack_ground_truth,
                "num_retrieved": len(retrieved_docs),
                "num_relevant": len(relevant_docs)
            }
            
            results["query_results"].append(query_result)
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(queries):
                logger.info(f"Processed {i + 1}/{len(queries)} queries")
        
        # Calculate overall retrieval metrics
        map_values = [r["retrieval_metrics"].get("map", 0.0) for r in results["query_results"]]
        mrr_values = [r["retrieval_metrics"].get("mrr", 0.0) for r in results["query_results"]]
        
        results["retrieval_metrics"]["mean_map"] = np.mean(map_values)
        results["retrieval_metrics"]["mean_mrr"] = np.mean(mrr_values)
        
        # Calculate precision@k and recall@k for different k values
        for k in [1, 5, 10, 20, 50]:
            precision_values = [
                r["retrieval_metrics"].get(f"precision@{k}", 0.0) 
                for r in results["query_results"]
            ]
            recall_values = [
                r["retrieval_metrics"].get(f"recall@{k}", 0.0) 
                for r in results["query_results"]
            ]
            
            results["retrieval_metrics"][f"mean_precision@{k}"] = np.mean(precision_values)
            results["retrieval_metrics"][f"mean_recall@{k}"] = np.mean(recall_values)
        
        # Calculate classification metrics
        results["classification_metrics"] = EvaluationMetrics.compute_classification_metrics(
            predictions=predictions,
            ground_truth=ground_truth_labels
        )
        
        # Calculate performance metrics
        results["performance_metrics"] = EvaluationMetrics.compute_performance_metrics(
            latencies=latencies,
            memory_usages=memory_usages
        )
        
        # Add system-specific metrics
        if system_name == "CacheRAG" or system_name == "HybridCacheGraphRAG":
            # Add cache metrics if available
            if hasattr(system, 'retriever_cache') and hasattr(system.retriever_cache, 'get_stats'):
                cache_stats = system.retriever_cache.get_stats()
                results["cache_metrics"] = EvaluationMetrics.compute_cache_metrics(
                    cache_hits=cache_stats.get("hits", 0),
                    cache_misses=cache_stats.get("misses", 0),
                    cache_size=cache_stats.get("size", 0),
                    cache_evictions=cache_stats.get("evictions", 0)
                )
        
        if system_name == "GraphRAG" or system_name == "HybridCacheGraphRAG":
            # Add graph metrics if available
            if hasattr(system, 'graph'):
                graph = system.graph
                node_types = {}
                edge_types = {}
                
                # Count node types
                for node_id, node_data in graph.graph.nodes(data=True):
                    node_type = node_data.get('type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                # Count edge types
                for _, _, edge_data in graph.graph.edges(data=True):
                    edge_type = edge_data.get('type', 'unknown')
                    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                
                results["graph_metrics"] = EvaluationMetrics.compute_graph_metrics(
                    graph_size=graph.graph.number_of_nodes(),
                    node_types=node_types,
                    edge_types=edge_types,
                    query_traversal_depth=0.0,  # Placeholder
                    traversal_time=0.0  # Placeholder
                )
        
        if system_name == "HybridCacheGraphRAG":
            # Add hybrid metrics if available
            if hasattr(system, 'retriever') and hasattr(system.retriever, 'get_performance_metrics'):
                hybrid_metrics = system.retriever.get_performance_metrics()
                results["hybrid_metrics"] = EvaluationMetrics.compute_hybrid_metrics(
                    cache_weight=system.retriever.cache_weight,
                    graph_weight=system.retriever.graph_weight,
                    synergy_factor=0.0,  # Placeholder
                    cache_hit_rate=hybrid_metrics.get("cache_hit_rate", 0.0),
                    graph_hit_rate=hybrid_metrics.get("graph_hit_rate", 0.0),
                    hybrid_hit_rate=hybrid_metrics.get("hybrid_hit_rate", 0.0)
                )
        
        # Save results
        self._save_results(results, system_name, test_scenario)
        
        return results
    
    def _save_results(
        self,
        results: Dict[str, Any],
        system_name: str,
        test_scenario: str
    ) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            results: Evaluation results
            system_name: Name of the system
            test_scenario: Name of the test scenario
        """
        # Create output directory
        system_dir = os.path.join(self.output_dir, system_name)
        os.makedirs(system_dir, exist_ok=True)
        
        # Save results to JSON file
        output_file = os.path.join(system_dir, f"{test_scenario}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_file}")
        
        # Store results in memory
        if system_name not in self.results:
            self.results[system_name] = {}
        
        self.results[system_name][test_scenario] = results
    
    def generate_comparison_report(
        self,
        system_names: List[str],
        test_scenarios: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a comparison report for multiple systems and test scenarios.
        
        Args:
            system_names: List of system names to compare
            test_scenarios: List of test scenarios to compare
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Generating comparison report for {len(system_names)} systems and {len(test_scenarios)} scenarios")
        
        comparison = {
            "systems": system_names,
            "scenarios": test_scenarios,
            "metrics": {},
            "rankings": {}
        }
        
        # Define metrics to compare
        metric_groups = {
            "retrieval": [
                "mean_map", "mean_mrr", "mean_precision@10", "mean_recall@10"
            ],
            "classification": [
                "accuracy", "precision", "recall", "f1"
            ],
            "performance": [
                "mean_latency", "median_latency", "p95_latency", "mean_memory"
            ]
        }
        
        # Collect metrics for each system and scenario
        for metric_group, metrics in metric_groups.items():
            comparison["metrics"][metric_group] = {}
            
            for metric in metrics:
                comparison["metrics"][metric_group][metric] = {}
                comparison["rankings"][metric] = {}
                
                for scenario in test_scenarios:
                    comparison["metrics"][metric_group][metric][scenario] = {}
                    comparison["rankings"][metric][scenario] = {}
                    
                    metric_values = []
                    
                    for system in system_names:
                        if system in self.results and scenario in self.results[system]:
                            result = self.results[system][scenario]
                            
                            # Extract metric value
                            if metric_group == "retrieval" and "retrieval_metrics" in result:
                                value = result["retrieval_metrics"].get(metric, 0.0)
                            elif metric_group == "classification" and "classification_metrics" in result:
                                value = result["classification_metrics"].get(metric, 0.0)
                            elif metric_group == "performance" and "performance_metrics" in result:
                                value = result["performance_metrics"].get(metric, 0.0)
                            else:
                                value = 0.0
                            
                            comparison["metrics"][metric_group][metric][scenario][system] = value
                            metric_values.append((system, value))
                    
                    # Rank systems for this metric and scenario
                    if metric_group in ["retrieval", "classification"]:
                        # Higher is better
                        ranked_systems = sorted(metric_values, key=lambda x: x[1], reverse=True)
                    else:
                        # Lower is better for performance metrics
                        ranked_systems = sorted(metric_values, key=lambda x: x[1])
                    
                    for rank, (system, _) in enumerate(ranked_systems):
                        comparison["rankings"][metric][scenario][system] = rank + 1
        
        # Calculate overall rankings
        comparison["overall_rankings"] = {}
        
        for scenario in test_scenarios:
            comparison["overall_rankings"][scenario] = {}
            
            for system in system_names:
                # Sum of ranks across all metrics
                rank_sum = 0
                count = 0
                
                for metric in comparison["rankings"]:
                    if scenario in comparison["rankings"][metric] and system in comparison["rankings"][metric][scenario]:
                        rank_sum += comparison["rankings"][metric][scenario][system]
                        count += 1
                
                if count > 0:
                    comparison["overall_rankings"][scenario][system] = rank_sum / count
                else:
                    comparison["overall_rankings"][scenario][system] = float('inf')
        
        # Save comparison report
        output_file = os.path.join(self.output_dir, "comparison_report.json")
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Saved comparison report to {output_file}")
        
        # Generate comparison charts
        self._generate_comparison_charts(comparison)
        
        return comparison
    
    def _generate_comparison_charts(self, comparison: Dict[str, Any]) -> None:
        """
        Generate comparison charts.
        
        Args:
            comparison: Comparison results
        """
        # Create charts directory
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Set style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Generate charts for each metric group
        for metric_group, metrics in comparison["metrics"].items():
            for metric, scenarios in metrics.items():
                for scenario, systems in scenarios.items():
                    # Create bar chart
                    plt.figure(figsize=(10, 6))
                    
                    # Sort systems by metric value
                    if metric_group in ["retrieval", "classification"]:
                        # Higher is better
                        sorted_systems = sorted(systems.items(), key=lambda x: x[1], reverse=True)
                    else:
                        # Lower is better for performance metrics
                        sorted_systems = sorted(systems.items(), key=lambda x: x[1])
                    
                    system_names = [s[0] for s in sorted_systems]
                    metric_values = [s[1] for s in sorted_systems]
                    
                    # Create bar chart
                    ax = sns.barplot(x=system_names, y=metric_values)
                    
                    # Add value labels
                    for i, v in enumerate(metric_values):
                        ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
                    
                    # Set labels and title
                    plt.xlabel("System")
                    plt.ylabel(metric)
                    plt.title(f"{metric} for {scenario} scenario")
                    
                    # Rotate x-axis labels
                    plt.xticks(rotation=45, ha='right')
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save chart
                    chart_file = os.path.join(charts_dir, f"{metric_group}_{metric}_{scenario}.png")
                    plt.savefig(chart_file)
                    plt.close()
        
        # Generate overall ranking chart
        for scenario, systems in comparison["overall_rankings"].items():
            plt.figure(figsize=(10, 6))
            
            # Sort systems by rank (lower is better)
            sorted_systems = sorted(systems.items(), key=lambda x: x[1])
            system_names = [s[0] for s in sorted_systems]
            ranks = [s[1] for s in sorted_systems]
            
            # Create bar chart
            ax = sns.barplot(x=system_names, y=ranks)
            
            # Add value labels
            for i, v in enumerate(ranks):
                ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
            
            # Set labels and title
            plt.xlabel("System")
            plt.ylabel("Average Rank")
            plt.title(f"Overall Rankings for {scenario} scenario")
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart
            chart_file = os.path.join(charts_dir, f"overall_ranking_{scenario}.png")
            plt.savefig(chart_file)
            plt.close()
        
        logger.info(f"Generated comparison charts in {charts_dir}")
    
    def perform_statistical_tests(
        self,
        system_names: List[str],
        test_scenarios: List[str],
        metric_groups: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform statistical significance tests.
        
        Args:
            system_names: List of system names to compare
            test_scenarios: List of test scenarios to compare
            metric_groups: List of metric groups to test
            
        Returns:
            Dictionary containing test results
        """
        from scipy import stats
        
        if metric_groups is None:
            metric_groups = ["retrieval", "classification", "performance"]
        
        logger.info(f"Performing statistical tests for {len(system_names)} systems and {len(test_scenarios)} scenarios")
        
        test_results = {
            "systems": system_names,
            "scenarios": test_scenarios,
            "metric_groups": metric_groups,
            "tests": {}
        }
        
        # Define metrics to test
        metrics_to_test = {
            "retrieval": ["map", "mrr", "precision@10", "recall@10"],
            "classification": ["accuracy", "precision", "recall", "f1"],
            "performance": ["latency", "memory_usage"]
        }
        
        # Perform tests for each scenario and metric
        for scenario in test_scenarios:
            test_results["tests"][scenario] = {}
            
            for metric_group in metric_groups:
                if metric_group not in metrics_to_test:
                    continue
                
                test_results["tests"][scenario][metric_group] = {}
                
                for metric in metrics_to_test[metric_group]:
                    test_results["tests"][scenario][metric_group][metric] = {}
                    
                    # Collect metric values for each system
                    system_values = {}
                    
                    for system in system_names:
                        if system in self.results and scenario in self.results[system]:
                            result = self.results[system][scenario]
                            
                            # Extract metric values from query results
                            values = []
                            
                            for query_result in result.get("query_results", []):
                                if metric_group == "retrieval":
                                    if metric in query_result.get("retrieval_metrics", {}):
                                        values.append(query_result["retrieval_metrics"][metric])
                                elif metric_group == "classification":
                                    # For classification, we need to compute per-query metrics
                                    # This is a simplification
                                    if query_result.get("is_attack_prediction") == query_result.get("is_attack_ground_truth"):
                                        values.append(1.0)  # Correct
                                    else:
                                        values.append(0.0)  # Incorrect
                                elif metric_group == "performance":
                                    if metric == "latency":
                                        values.append(query_result.get("latency", 0.0))
                                    elif metric == "memory_usage":
                                        values.append(query_result.get("memory_usage", 0.0))
                            
                            if values:
                                system_values[system] = values
                    
                    # Perform pairwise t-tests
                    for i, system1 in enumerate(system_names):
                        for system2 in system_names[i+1:]:
                            if system1 in system_values and system2 in system_values:
                                values1 = system_values[system1]
                                values2 = system_values[system2]
                                
                                # Perform t-test
                                t_stat, p_value = stats.ttest_ind(values1, values2)
                                
                                # Determine if difference is significant
                                alpha = 0.05
                                is_significant = p_value < alpha
                                
                                # Determine which system is better
                                if metric_group in ["retrieval", "classification"]:
                                    # Higher is better
                                    better_system = system1 if np.mean(values1) > np.mean(values2) else system2
                                else:
                                    # Lower is better for performance metrics
                                    better_system = system1 if np.mean(values1) < np.mean(values2) else system2
                                
                                # Store test result
                                test_key = f"{system1}_vs_{system2}"
                                test_results["tests"][scenario][metric_group][metric][test_key] = {
                                    "t_statistic": float(t_stat),
                                    "p_value": float(p_value),
                                    "is_significant": is_significant,
                                    "better_system": better_system if is_significant else "no significant difference",
                                    "mean_1": float(np.mean(values1)),
                                    "mean_2": float(np.mean(values2)),
                                    "std_1": float(np.std(values1)),
                                    "std_2": float(np.std(values2))
                                }
        
        # Save test results
        output_file = os.path.join(self.output_dir, "statistical_tests.json")
        with open(output_file, "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Saved statistical test results to {output_file}")
        
        return test_results
