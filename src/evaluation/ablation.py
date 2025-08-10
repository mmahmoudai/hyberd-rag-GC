"""
Ablation study framework for RAG systems in network security packet analysis.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ..common.base import RAGSystem
from ..traditional_rag.traditional_rag import TraditionalRAG, NetworkEmbeddingModel, VectorRetriever, NetworkSecurityGenerator
from ..graph_rag.graph_rag import GraphRAG, NetworkGraph, GraphRetriever
from ..cache_rag.cache_rag import CacheRAG, CacheSystem, CachedRetriever
from ..hybrid_rag.hybrid_rag import HybridCacheGraphRAG, HybridRetriever
from .evaluation_metrics import RAGEvaluator
from .test_scenarios import BenchmarkingFramework

# Configure logging
logger = logging.getLogger(__name__)

class AblationStudy:
    """
    Class for conducting ablation studies on RAG systems.
    """
    def __init__(
        self,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the ablation study.
        
        Args:
            output_dir: Directory to save ablation study results
        """
        self.output_dir = output_dir or Path("/home/ubuntu/research/experiments/ablation")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize benchmarking framework
        self.framework = BenchmarkingFramework()
        self.framework.load_scenarios()
        
        # Initialize evaluator
        self.evaluator = RAGEvaluator(output_dir=self.output_dir)
        
        # Results storage
        self.results = {}
    
    def run_traditional_rag_ablation(
        self,
        scenario_name: str,
        embedding_dims: List[int] = [64, 128, 256],
        chunk_sizes: List[int] = [1, 5, 10],
        retrieval_methods: List[str] = ["cosine", "dot", "euclidean"]
    ) -> Dict[str, Any]:
        """
        Run ablation study on Traditional RAG.
        
        Args:
            scenario_name: Name of the scenario to use
            embedding_dims: List of embedding dimensions to test
            chunk_sizes: List of chunk sizes to test
            retrieval_methods: List of retrieval methods to test
            
        Returns:
            Dictionary containing ablation study results
        """
        logger.info(f"Running Traditional RAG ablation study on {scenario_name}")
        
        # Prepare queries and ground truth
        queries, ground_truth = self.framework.prepare_queries(scenario_name)
        
        if not queries:
            logger.warning(f"No queries found for scenario: {scenario_name}")
            return {}
        
        # Initialize results
        results = {
            "scenario": scenario_name,
            "system": "TraditionalRAG",
            "variants": {},
            "baseline": None
        }
        
        # Run baseline
        baseline_system = TraditionalRAG()
        baseline_results = self.evaluator.evaluate_system(
            system=baseline_system,
            system_name="TraditionalRAG_Baseline",
            queries=queries,
            ground_truth=ground_truth,
            test_scenario=scenario_name
        )
        
        results["baseline"] = baseline_results
        
        # Test embedding dimensions
        for dim in embedding_dims:
            variant_name = f"TraditionalRAG_Embed{dim}"
            
            # Create custom embedding model
            embedding_model = NetworkEmbeddingModel(embedding_dim=dim)
            
            # Create system with custom embedding model
            system = TraditionalRAG(embedding_model=embedding_model)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test chunk sizes
        for chunk_size in chunk_sizes:
            variant_name = f"TraditionalRAG_Chunk{chunk_size}"
            
            # Create system with custom chunk size
            system = TraditionalRAG(chunk_size=chunk_size)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test retrieval methods
        for method in retrieval_methods:
            variant_name = f"TraditionalRAG_Retrieval{method.capitalize()}"
            
            # Create embedding model
            embedding_model = NetworkEmbeddingModel()
            
            # Create retriever with custom method
            retriever = VectorRetriever(
                embedding_model=embedding_model,
                similarity_metric=method
            )
            
            # Create system with custom retriever
            system = TraditionalRAG(
                embedding_model=embedding_model,
                retriever=retriever
            )
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Save results
        self._save_results("traditional_rag", scenario_name, results)
        
        return results
    
    def run_graph_rag_ablation(
        self,
        scenario_name: str,
        graph_types: List[str] = ["directed", "undirected"],
        node_types: List[List[str]] = [["ip", "port", "protocol"], ["ip", "port"], ["ip"]],
        traversal_depths: List[int] = [1, 2, 3]
    ) -> Dict[str, Any]:
        """
        Run ablation study on Graph RAG.
        
        Args:
            scenario_name: Name of the scenario to use
            graph_types: List of graph types to test
            node_types: List of node type combinations to test
            traversal_depths: List of traversal depths to test
            
        Returns:
            Dictionary containing ablation study results
        """
        logger.info(f"Running Graph RAG ablation study on {scenario_name}")
        
        # Prepare queries and ground truth
        queries, ground_truth = self.framework.prepare_queries(scenario_name)
        
        if not queries:
            logger.warning(f"No queries found for scenario: {scenario_name}")
            return {}
        
        # Initialize results
        results = {
            "scenario": scenario_name,
            "system": "GraphRAG",
            "variants": {},
            "baseline": None
        }
        
        # Run baseline
        baseline_system = GraphRAG()
        baseline_results = self.evaluator.evaluate_system(
            system=baseline_system,
            system_name="GraphRAG_Baseline",
            queries=queries,
            ground_truth=ground_truth,
            test_scenario=scenario_name
        )
        
        results["baseline"] = baseline_results
        
        # Test graph types
        for graph_type in graph_types:
            variant_name = f"GraphRAG_Graph{graph_type.capitalize()}"
            
            # Create custom graph
            graph = NetworkGraph(directed=(graph_type == "directed"))
            
            # Create system with custom graph
            system = GraphRAG(graph=graph)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test node types
        for node_type_list in node_types:
            variant_name = f"GraphRAG_Nodes{''.join(nt.capitalize() for nt in node_type_list)}"
            
            # Create custom graph
            graph = NetworkGraph(node_types=node_type_list)
            
            # Create system with custom graph
            system = GraphRAG(graph=graph)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test traversal depths
        for depth in traversal_depths:
            variant_name = f"GraphRAG_Depth{depth}"
            
            # Create custom retriever
            graph = NetworkGraph()
            retriever = GraphRetriever(graph=graph, max_traversal_depth=depth)
            
            # Create system with custom retriever
            system = GraphRAG(graph=graph, retriever=retriever)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Save results
        self._save_results("graph_rag", scenario_name, results)
        
        return results
    
    def run_cache_rag_ablation(
        self,
        scenario_name: str,
        cache_sizes: List[int] = [100, 500, 1000],
        ttl_values: List[int] = [60, 300, 600],
        policies: List[str] = ["LRU", "LFU", "TLFU"]
    ) -> Dict[str, Any]:
        """
        Run ablation study on Cache RAG.
        
        Args:
            scenario_name: Name of the scenario to use
            cache_sizes: List of cache sizes to test
            ttl_values: List of TTL values to test
            policies: List of cache policies to test
            
        Returns:
            Dictionary containing ablation study results
        """
        logger.info(f"Running Cache RAG ablation study on {scenario_name}")
        
        # Prepare queries and ground truth
        queries, ground_truth = self.framework.prepare_queries(scenario_name)
        
        if not queries:
            logger.warning(f"No queries found for scenario: {scenario_name}")
            return {}
        
        # Initialize results
        results = {
            "scenario": scenario_name,
            "system": "CacheRAG",
            "variants": {},
            "baseline": None
        }
        
        # Run baseline
        baseline_system = CacheRAG()
        baseline_results = self.evaluator.evaluate_system(
            system=baseline_system,
            system_name="CacheRAG_Baseline",
            queries=queries,
            ground_truth=ground_truth,
            test_scenario=scenario_name
        )
        
        results["baseline"] = baseline_results
        
        # Test cache sizes
        for size in cache_sizes:
            variant_name = f"CacheRAG_Size{size}"
            
            # Create custom cache system
            cache_system = CacheSystem(max_size=size)
            
            # Create system with custom cache
            system = CacheRAG(retriever_cache=cache_system)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test TTL values
        for ttl in ttl_values:
            variant_name = f"CacheRAG_TTL{ttl}"
            
            # Create custom cache system
            cache_system = CacheSystem(ttl=ttl)
            
            # Create system with custom cache
            system = CacheRAG(retriever_cache=cache_system)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test cache policies
        for policy in policies:
            variant_name = f"CacheRAG_Policy{policy}"
            
            # Create custom cache system
            cache_system = CacheSystem(policy=policy)
            
            # Create system with custom cache
            system = CacheRAG(retriever_cache=cache_system)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Save results
        self._save_results("cache_rag", scenario_name, results)
        
        return results
    
    def run_hybrid_rag_ablation(
        self,
        scenario_name: str,
        cache_weights: List[float] = [0.3, 0.5, 0.7],
        graph_weights: List[float] = [0.7, 0.5, 0.3],
        adaptive_modes: List[bool] = [True, False]
    ) -> Dict[str, Any]:
        """
        Run ablation study on Hybrid Cache-Graph RAG.
        
        Args:
            scenario_name: Name of the scenario to use
            cache_weights: List of cache weights to test
            graph_weights: List of graph weights to test
            adaptive_modes: List of adaptive mode settings to test
            
        Returns:
            Dictionary containing ablation study results
        """
        logger.info(f"Running Hybrid Cache-Graph RAG ablation study on {scenario_name}")
        
        # Prepare queries and ground truth
        queries, ground_truth = self.framework.prepare_queries(scenario_name)
        
        if not queries:
            logger.warning(f"No queries found for scenario: {scenario_name}")
            return {}
        
        # Initialize results
        results = {
            "scenario": scenario_name,
            "system": "HybridCacheGraphRAG",
            "variants": {},
            "baseline": None
        }
        
        # Run baseline
        baseline_system = HybridCacheGraphRAG()
        baseline_results = self.evaluator.evaluate_system(
            system=baseline_system,
            system_name="HybridCacheGraphRAG_Baseline",
            queries=queries,
            ground_truth=ground_truth,
            test_scenario=scenario_name
        )
        
        results["baseline"] = baseline_results
        
        # Test weight combinations
        for cache_weight, graph_weight in zip(cache_weights, graph_weights):
            variant_name = f"HybridRAG_Cache{int(cache_weight*10)}Graph{int(graph_weight*10)}"
            
            # Create system with custom weights
            system = HybridCacheGraphRAG(
                cache_weight=cache_weight,
                graph_weight=graph_weight
            )
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test adaptive modes
        for adaptive in adaptive_modes:
            variant_name = f"HybridRAG_Adaptive{str(adaptive)}"
            
            # Create system with custom adaptive setting
            system = HybridCacheGraphRAG(adaptive=adaptive)
            
            # Evaluate system
            variant_results = self.evaluator.evaluate_system(
                system=system,
                system_name=variant_name,
                queries=queries,
                ground_truth=ground_truth,
                test_scenario=scenario_name
            )
            
            results["variants"][variant_name] = variant_results
        
        # Test component removal (cache only)
        variant_name = "HybridRAG_CacheOnly"
        
        # Create system with graph weight = 0
        system = HybridCacheGraphRAG(
            cache_weight=1.0,
            graph_weight=0.0
        )
        
        # Evaluate system
        variant_results = self.evaluator.evaluate_system(
            system=system,
            system_name=variant_name,
            queries=queries,
            ground_truth=ground_truth,
            test_scenario=scenario_name
        )
        
        results["variants"][variant_name] = variant_results
        
        # Test component removal (graph only)
        variant_name = "HybridRAG_GraphOnly"
        
        # Create system with cache weight = 0
        system = HybridCacheGraphRAG(
            cache_weight=0.0,
            graph_weight=1.0
        )
        
        # Evaluate system
        variant_results = self.evaluator.evaluate_system(
            system=system,
            system_name=variant_name,
            queries=queries,
            ground_truth=ground_truth,
            test_scenario=scenario_name
        )
        
        results["variants"][variant_name] = variant_results
        
        # Save results
        self._save_results("hybrid_rag", scenario_name, results)
        
        return results
    
    def _save_results(
        self,
        system_type: str,
        scenario_name: str,
        results: Dict[str, Any]
    ) -> None:
        """
        Save ablation study results.
        
        Args:
            system_type: Type of RAG system
            scenario_name: Name of the scenario
            results: Ablation study results
        """
        # Create output directory
        system_dir = os.path.join(self.output_dir, system_type)
        os.makedirs(system_dir, exist_ok=True)
        
        # Save results to JSON file
        output_file = os.path.join(system_dir, f"{scenario_name}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved ablation study results to {output_file}")
        
        # Store results in memory
        if system_type not in self.results:
            self.results[system_type] = {}
        
        self.results[system_type][scenario_name] = results
    
    def generate_ablation_report(self) -> None:
        """Generate a report of ablation study results."""
        # Create report directory
        report_dir = os.path.join(self.output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Process each system type
        for system_type, scenarios in self.results.items():
            for scenario_name, results in scenarios.items():
                self._generate_system_ablation_charts(
                    system_type=system_type,
                    scenario_name=scenario_name,
                    results=results,
                    output_dir=report_dir
                )
        
        # Generate summary report
        self._generate_summary_report(output_dir=report_dir)
        
        logger.info(f"Generated ablation study report in {report_dir}")
    
    def _generate_system_ablation_charts(
        self,
        system_type: str,
        scenario_name: str,
        results: Dict[str, Any],
        output_dir: str
    ) -> None:
        """
        Generate charts for a system's ablation study.
        
        Args:
            system_type: Type of RAG system
            scenario_name: Name of the scenario
            results: Ablation study results
            output_dir: Directory to save charts
        """
        # Extract baseline and variants
        baseline = results.get("baseline", {})
        variants = results.get("variants", {})
        
        if not baseline or not variants:
            logger.warning(f"No baseline or variants found for {system_type} on {scenario_name}")
            return
        
        # Create directory for this system and scenario
        chart_dir = os.path.join(output_dir, system_type, scenario_name)
        os.makedirs(chart_dir, exist_ok=True)
        
        # Define metrics to chart
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
        
        # Generate charts for each metric group
        for group_name, metrics in metric_groups.items():
            for metric in metrics:
                # Extract baseline value
                if group_name == "retrieval":
                    baseline_value = baseline.get("retrieval_metrics", {}).get(metric, 0.0)
                elif group_name == "classification":
                    baseline_value = baseline.get("classification_metrics", {}).get(metric, 0.0)
                elif group_name == "performance":
                    baseline_value = baseline.get("performance_metrics", {}).get(metric, 0.0)
                else:
                    baseline_value = 0.0
                
                # Extract variant values
                variant_values = {}
                
                for variant_name, variant_results in variants.items():
                    if group_name == "retrieval":
                        value = variant_results.get("retrieval_metrics", {}).get(metric, 0.0)
                    elif group_name == "classification":
                        value = variant_results.get("classification_metrics", {}).get(metric, 0.0)
                    elif group_name == "performance":
                        value = variant_results.get("performance_metrics", {}).get(metric, 0.0)
                    else:
                        value = 0.0
                    
                    variant_values[variant_name] = value
                
                # Create chart
                plt.figure(figsize=(12, 6))
                
                # Sort variants by value
                if group_name in ["retrieval", "classification"]:
                    # Higher is better
                    sorted_variants = sorted(
                        variant_values.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                else:
                    # Lower is better for performance metrics
                    sorted_variants = sorted(
                        variant_values.items(),
                        key=lambda x: x[1]
                    )
                
                # Extract names and values
                variant_names = [v[0] for v in sorted_variants]
                values = [v[1] for v in sorted_variants]
                
                # Create bar chart
                bars = plt.bar(variant_names, values)
                
                # Add baseline line
                plt.axhline(
                    y=baseline_value,
                    color='r',
                    linestyle='-',
                    label=f'Baseline: {baseline_value:.3f}'
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom'
                    )
                
                # Set labels and title
                plt.xlabel('Variants')
                plt.ylabel(metric)
                plt.title(f'{metric} for {system_type} on {scenario_name}')
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                
                # Save chart
                chart_file = os.path.join(chart_dir, f"{group_name}_{metric}.png")
                plt.savefig(chart_file)
                plt.close()
    
    def _generate_summary_report(self, output_dir: str) -> None:
        """
        Generate a summary report of ablation study results.
        
        Args:
            output_dir: Directory to save the report
        """
        # Create summary data
        summary = {
            "systems": list(self.results.keys()),
            "scenarios": [],
            "best_variants": {},
            "component_impact": {}
        }
        
        # Collect scenarios
        for system_type, scenarios in self.results.items():
            summary["scenarios"].extend(list(scenarios.keys()))
        
        summary["scenarios"] = sorted(list(set(summary["scenarios"])))
        
        # Find best variants for each system and scenario
        for system_type, scenarios in self.results.items():
            summary["best_variants"][system_type] = {}
            
            for scenario_name, results in scenarios.items():
                baseline = results.get("baseline", {})
                variants = results.get("variants", {})
                
                if not baseline or not variants:
                    continue
                
                # Find best variant for retrieval
                best_retrieval_variant = None
                best_retrieval_value = 0.0
                
                for variant_name, variant_results in variants.items():
                    value = variant_results.get("retrieval_metrics", {}).get("mean_map", 0.0)
                    
                    if value > best_retrieval_value:
                        best_retrieval_value = value
                        best_retrieval_variant = variant_name
                
                # Find best variant for classification
                best_classification_variant = None
                best_classification_value = 0.0
                
                for variant_name, variant_results in variants.items():
                    value = variant_results.get("classification_metrics", {}).get("f1", 0.0)
                    
                    if value > best_classification_value:
                        best_classification_value = value
                        best_classification_variant = variant_name
                
                # Find best variant for performance
                best_performance_variant = None
                best_performance_value = float('inf')
                
                for variant_name, variant_results in variants.items():
                    value = variant_results.get("performance_metrics", {}).get("mean_latency", float('inf'))
                    
                    if value < best_performance_value:
                        best_performance_value = value
                        best_performance_variant = variant_name
                
                # Store best variants
                summary["best_variants"][system_type][scenario_name] = {
                    "retrieval": {
                        "variant": best_retrieval_variant,
                        "value": best_retrieval_value,
                        "baseline": baseline.get("retrieval_metrics", {}).get("mean_map", 0.0),
                        "improvement": best_retrieval_value - baseline.get("retrieval_metrics", {}).get("mean_map", 0.0)
                    },
                    "classification": {
                        "variant": best_classification_variant,
                        "value": best_classification_value,
                        "baseline": baseline.get("classification_metrics", {}).get("f1", 0.0),
                        "improvement": best_classification_value - baseline.get("classification_metrics", {}).get("f1", 0.0)
                    },
                    "performance": {
                        "variant": best_performance_variant,
                        "value": best_performance_value,
                        "baseline": baseline.get("performance_metrics", {}).get("mean_latency", 0.0),
                        "improvement": baseline.get("performance_metrics", {}).get("mean_latency", 0.0) - best_performance_value
                    }
                }
        
        # Analyze component impact for hybrid system
        if "hybrid_rag" in self.results:
            summary["component_impact"]["hybrid_rag"] = {}
            
            for scenario_name, results in self.results["hybrid_rag"].items():
                baseline = results.get("baseline", {})
                variants = results.get("variants", {})
                
                if not baseline or not variants:
                    continue
                
                # Get cache-only and graph-only variants
                cache_only = variants.get("HybridRAG_CacheOnly", {})
                graph_only = variants.get("HybridRAG_GraphOnly", {})
                
                if not cache_only or not graph_only:
                    continue
                
                # Calculate impact on retrieval
                baseline_retrieval = baseline.get("retrieval_metrics", {}).get("mean_map", 0.0)
                cache_retrieval = cache_only.get("retrieval_metrics", {}).get("mean_map", 0.0)
                graph_retrieval = graph_only.get("retrieval_metrics", {}).get("mean_map", 0.0)
                
                # Calculate impact on classification
                baseline_classification = baseline.get("classification_metrics", {}).get("f1", 0.0)
                cache_classification = cache_only.get("classification_metrics", {}).get("f1", 0.0)
                graph_classification = graph_only.get("classification_metrics", {}).get("f1", 0.0)
                
                # Calculate impact on performance
                baseline_performance = baseline.get("performance_metrics", {}).get("mean_latency", 0.0)
                cache_performance = cache_only.get("performance_metrics", {}).get("mean_latency", 0.0)
                graph_performance = graph_only.get("performance_metrics", {}).get("mean_latency", 0.0)
                
                # Store component impact
                summary["component_impact"]["hybrid_rag"][scenario_name] = {
                    "retrieval": {
                        "baseline": baseline_retrieval,
                        "cache_only": cache_retrieval,
                        "graph_only": graph_retrieval,
                        "synergy": baseline_retrieval - max(cache_retrieval, graph_retrieval)
                    },
                    "classification": {
                        "baseline": baseline_classification,
                        "cache_only": cache_classification,
                        "graph_only": graph_classification,
                        "synergy": baseline_classification - max(cache_classification, graph_classification)
                    },
                    "performance": {
                        "baseline": baseline_performance,
                        "cache_only": cache_performance,
                        "graph_only": graph_performance,
                        "synergy": min(cache_performance, graph_performance) - baseline_performance
                    }
                }
        
        # Save summary report
        summary_file = os.path.join(output_dir, "ablation_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate summary charts
        self._generate_summary_charts(summary, output_dir)
    
    def _generate_summary_charts(
        self,
        summary: Dict[str, Any],
        output_dir: str
    ) -> None:
        """
        Generate summary charts for ablation study.
        
        Args:
            summary: Summary data
            output_dir: Directory to save charts
        """
        # Create improvement charts for each system
        for system_type in summary["systems"]:
            if system_type not in summary["best_variants"]:
                continue
            
            # Create retrieval improvement chart
            plt.figure(figsize=(10, 6))
            
            scenarios = []
            improvements = []
            
            for scenario_name, metrics in summary["best_variants"][system_type].items():
                scenarios.append(scenario_name)
                improvements.append(metrics["retrieval"]["improvement"])
            
            plt.bar(scenarios, improvements)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Scenarios')
            plt.ylabel('MAP Improvement')
            plt.title(f'Retrieval Improvement for {system_type}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_file = os.path.join(output_dir, f"{system_type}_retrieval_improvement.png")
            plt.savefig(chart_file)
            plt.close()
            
            # Create classification improvement chart
            plt.figure(figsize=(10, 6))
            
            scenarios = []
            improvements = []
            
            for scenario_name, metrics in summary["best_variants"][system_type].items():
                scenarios.append(scenario_name)
                improvements.append(metrics["classification"]["improvement"])
            
            plt.bar(scenarios, improvements)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Scenarios')
            plt.ylabel('F1 Improvement')
            plt.title(f'Classification Improvement for {system_type}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_file = os.path.join(output_dir, f"{system_type}_classification_improvement.png")
            plt.savefig(chart_file)
            plt.close()
            
            # Create performance improvement chart
            plt.figure(figsize=(10, 6))
            
            scenarios = []
            improvements = []
            
            for scenario_name, metrics in summary["best_variants"][system_type].items():
                scenarios.append(scenario_name)
                improvements.append(metrics["performance"]["improvement"])
            
            plt.bar(scenarios, improvements)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Scenarios')
            plt.ylabel('Latency Improvement (s)')
            plt.title(f'Performance Improvement for {system_type}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_file = os.path.join(output_dir, f"{system_type}_performance_improvement.png")
            plt.savefig(chart_file)
            plt.close()
        
        # Create component impact charts for hybrid system
        if "hybrid_rag" in summary["component_impact"]:
            for scenario_name, metrics in summary["component_impact"]["hybrid_rag"].items():
                # Create retrieval impact chart
                plt.figure(figsize=(8, 6))
                
                components = ["Baseline", "Cache Only", "Graph Only"]
                values = [
                    metrics["retrieval"]["baseline"],
                    metrics["retrieval"]["cache_only"],
                    metrics["retrieval"]["graph_only"]
                ]
                
                plt.bar(components, values)
                plt.xlabel('Components')
                plt.ylabel('MAP')
                plt.title(f'Component Impact on Retrieval for {scenario_name}')
                plt.tight_layout()
                
                chart_file = os.path.join(output_dir, f"hybrid_component_retrieval_{scenario_name}.png")
                plt.savefig(chart_file)
                plt.close()
                
                # Create classification impact chart
                plt.figure(figsize=(8, 6))
                
                components = ["Baseline", "Cache Only", "Graph Only"]
                values = [
                    metrics["classification"]["baseline"],
                    metrics["classification"]["cache_only"],
                    metrics["classification"]["graph_only"]
                ]
                
                plt.bar(components, values)
                plt.xlabel('Components')
                plt.ylabel('F1 Score')
                plt.title(f'Component Impact on Classification for {scenario_name}')
                plt.tight_layout()
                
                chart_file = os.path.join(output_dir, f"hybrid_component_classification_{scenario_name}.png")
                plt.savefig(chart_file)
                plt.close()
                
                # Create performance impact chart
                plt.figure(figsize=(8, 6))
                
                components = ["Baseline", "Cache Only", "Graph Only"]
                values = [
                    metrics["performance"]["baseline"],
                    metrics["performance"]["cache_only"],
                    metrics["performance"]["graph_only"]
                ]
                
                plt.bar(components, values)
                plt.xlabel('Components')
                plt.ylabel('Mean Latency (s)')
                plt.title(f'Component Impact on Performance for {scenario_name}')
                plt.tight_layout()
                
                chart_file = os.path.join(output_dir, f"hybrid_component_performance_{scenario_name}.png")
                plt.savefig(chart_file)
                plt.close()
                
                # Create synergy chart
                plt.figure(figsize=(8, 6))
                
                metrics_list = ["Retrieval", "Classification", "Performance"]
                synergy_values = [
                    metrics["retrieval"]["synergy"],
                    metrics["classification"]["synergy"],
                    metrics["performance"]["synergy"]
                ]
                
                plt.bar(metrics_list, synergy_values)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('Metrics')
                plt.ylabel('Synergy Effect')
                plt.title(f'Component Synergy for {scenario_name}')
                plt.tight_layout()
                
                chart_file = os.path.join(output_dir, f"hybrid_synergy_{scenario_name}.png")
                plt.savefig(chart_file)
                plt.close()

def run_ablation_studies(
    scenarios: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run ablation studies on all RAG systems.
    
    Args:
        scenarios: List of scenario names to use (defaults to all)
        output_dir: Directory to save ablation study results
    
    Returns:
        Dictionary containing ablation study results
    """
    # Initialize ablation study
    study = AblationStudy(output_dir=output_dir)
    
    # Determine scenarios to use
    if scenarios is None:
        scenarios = list(study.framework.scenarios.keys())
    else:
        # Validate scenarios
        for scenario in scenarios:
            if scenario not in study.framework.scenarios:
                logger.warning(f"Unknown scenario: {scenario}")
                scenarios.remove(scenario)
    
    if not scenarios:
        logger.error("No valid scenarios for ablation study")
        return {}
    
    # Run ablation studies for each system and scenario
    for scenario in scenarios:
        logger.info(f"Running ablation studies for scenario: {scenario}")
        
        # Traditional RAG
        try:
            study.run_traditional_rag_ablation(scenario)
        except Exception as e:
            logger.error(f"Error in Traditional RAG ablation for {scenario}: {e}")
        
        # Graph RAG
        try:
            study.run_graph_rag_ablation(scenario)
        except Exception as e:
            logger.error(f"Error in Graph RAG ablation for {scenario}: {e}")
        
        # Cache RAG
        try:
            study.run_cache_rag_ablation(scenario)
        except Exception as e:
            logger.error(f"Error in Cache RAG ablation for {scenario}: {e}")
        
        # Hybrid Cache-Graph RAG
        try:
            study.run_hybrid_rag_ablation(scenario)
        except Exception as e:
            logger.error(f"Error in Hybrid RAG ablation for {scenario}: {e}")
    
    # Generate ablation report
    study.generate_ablation_report()
    
    return study.results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("/home/ubuntu/research/experiments/ablation.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run ablation studies
    run_ablation_studies()
