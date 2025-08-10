"""
Main benchmarking script for comparing RAG systems in network security packet analysis.
"""

import os
import logging
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ..common.base import RAGSystem
from ..traditional_rag.traditional_rag import TraditionalRAG
from ..graph_rag.graph_rag import GraphRAG
from ..cache_rag.cache_rag import CacheRAG
from ..hybrid_rag.hybrid_rag import HybridCacheGraphRAG
from .dataset_processing import CICIDSProcessor, UNSWProcessor, CustomPCAPProcessor
from .evaluation_metrics import RAGEvaluator
from .test_scenarios import BenchmarkingFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/research/experiments/benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_rag_systems() -> Dict[str, RAGSystem]:
    """
    Initialize all RAG systems for benchmarking.
    
    Returns:
        Dictionary mapping system names to RAG system instances
    """
    logger.info("Initializing RAG systems")
    
    systems = {}
    
    # Initialize Traditional RAG
    try:
        traditional_rag = TraditionalRAG()
        systems["TraditionalRAG"] = traditional_rag
        logger.info("Initialized TraditionalRAG")
    except Exception as e:
        logger.error(f"Failed to initialize TraditionalRAG: {e}")
    
    # Initialize Graph RAG
    try:
        graph_rag = GraphRAG()
        systems["GraphRAG"] = graph_rag
        logger.info("Initialized GraphRAG")
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {e}")
    
    # Initialize Cache RAG
    try:
        cache_rag = CacheRAG()
        systems["CacheRAG"] = cache_rag
        logger.info("Initialized CacheRAG")
    except Exception as e:
        logger.error(f"Failed to initialize CacheRAG: {e}")
    
    # Initialize Hybrid Cache-Graph RAG
    try:
        hybrid_rag = HybridCacheGraphRAG()
        systems["HybridCacheGraphRAG"] = hybrid_rag
        logger.info("Initialized HybridCacheGraphRAG")
    except Exception as e:
        logger.error(f"Failed to initialize HybridCacheGraphRAG: {e}")
    
    return systems

def load_training_data(systems: Dict[str, RAGSystem]) -> None:
    """
    Load training data into all RAG systems.
    
    Args:
        systems: Dictionary mapping system names to RAG system instances
    """
    logger.info("Loading training data into RAG systems")
    
    # Load datasets
    cic_processor = CICIDSProcessor()
    unsw_processor = UNSWProcessor()
    
    # Process datasets
    cic_splits = cic_processor.process()
    unsw_splits = unsw_processor.process()
    
    # Use training splits
    cic_train_docs = cic_splits.get("train", [])
    unsw_train_docs = unsw_splits.get("train", [])
    
    # Combine documents
    all_docs = cic_train_docs + unsw_train_docs
    
    if not all_docs:
        logger.error("No training documents found")
        return
    
    logger.info(f"Loading {len(all_docs)} training documents into RAG systems")
    
    # Load documents into each system
    for system_name, system in systems.items():
        try:
            system.add_documents(all_docs)
            logger.info(f"Loaded training data into {system_name}")
        except Exception as e:
            logger.error(f"Failed to load training data into {system_name}: {e}")

def run_benchmark(
    systems: Dict[str, RAGSystem],
    scenarios: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run benchmark on all RAG systems.
    
    Args:
        systems: Dictionary mapping system names to RAG system instances
        scenarios: List of scenario names to benchmark (defaults to all)
        output_dir: Directory to save benchmark results
    
    Returns:
        Dictionary containing benchmark results
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path("/home/ubuntu/research/experiments/results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize benchmarking framework
    framework = BenchmarkingFramework()
    framework.load_scenarios()
    
    # Determine scenarios to benchmark
    if scenarios is None:
        scenarios = list(framework.scenarios.keys())
    else:
        # Validate scenarios
        for scenario in scenarios:
            if scenario not in framework.scenarios:
                logger.warning(f"Unknown scenario: {scenario}")
                scenarios.remove(scenario)
    
    if not scenarios:
        logger.error("No valid scenarios to benchmark")
        return {}
    
    # Save benchmark configuration
    framework.save_benchmark_config(
        system_names=list(systems.keys()),
        scenario_names=scenarios
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator(output_dir=output_dir)
    
    # Run benchmark for each system and scenario
    results = {}
    
    for scenario_name in scenarios:
        logger.info(f"Running benchmark for scenario: {scenario_name}")
        
        # Prepare queries and ground truth
        queries, ground_truth = framework.prepare_queries(scenario_name)
        
        if not queries:
            logger.warning(f"No queries found for scenario: {scenario_name}")
            continue
        
        logger.info(f"Prepared {len(queries)} queries for scenario: {scenario_name}")
        
        # Evaluate each system
        for system_name, system in systems.items():
            logger.info(f"Evaluating {system_name} on {scenario_name}")
            
            try:
                # Clear caches if applicable
                if hasattr(system, 'clear_caches'):
                    system.clear_caches()
                
                # Evaluate system
                system_results = evaluator.evaluate_system(
                    system=system,
                    system_name=system_name,
                    queries=queries,
                    ground_truth=ground_truth,
                    test_scenario=scenario_name
                )
                
                # Store results
                if system_name not in results:
                    results[system_name] = {}
                
                results[system_name][scenario_name] = system_results
                
                logger.info(f"Completed evaluation of {system_name} on {scenario_name}")
            except Exception as e:
                logger.error(f"Error evaluating {system_name} on {scenario_name}: {e}")
    
    # Generate comparison report
    try:
        comparison = evaluator.generate_comparison_report(
            system_names=list(systems.keys()),
            test_scenarios=scenarios
        )
        
        # Store comparison in results
        results["comparison"] = comparison
        
        logger.info("Generated comparison report")
    except Exception as e:
        logger.error(f"Error generating comparison report: {e}")
    
    # Perform statistical tests
    try:
        test_results = evaluator.perform_statistical_tests(
            system_names=list(systems.keys()),
            test_scenarios=scenarios
        )
        
        # Store test results in results
        results["statistical_tests"] = test_results
        
        logger.info("Performed statistical significance tests")
    except Exception as e:
        logger.error(f"Error performing statistical tests: {e}")
    
    # Save overall results
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved benchmark results to {results_file}")
    
    return results

def generate_summary_report(
    results: Dict[str, Any],
    output_dir: Optional[str] = None
) -> None:
    """
    Generate a summary report of benchmark results.
    
    Args:
        results: Dictionary containing benchmark results
        output_dir: Directory to save the report
    """
    if output_dir is None:
        output_dir = Path("/home/ubuntu/research/experiments/results")
    
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Extract system names and scenarios
    system_names = [name for name in results.keys() if name not in ["comparison", "statistical_tests"]]
    scenarios = []
    
    for system_name in system_names:
        scenarios.extend(list(results[system_name].keys()))
    
    scenarios = sorted(list(set(scenarios)))
    
    if not system_names or not scenarios:
        logger.error("No valid results to generate report")
        return
    
    # Create summary tables
    tables = {}
    
    # Retrieval metrics table
    retrieval_metrics = ["mean_map", "mean_mrr", "mean_precision@10", "mean_recall@10"]
    retrieval_table = pd.DataFrame(index=system_names, columns=scenarios)
    
    for metric in retrieval_metrics:
        table = pd.DataFrame(index=system_names, columns=scenarios)
        
        for system_name in system_names:
            for scenario in scenarios:
                if scenario in results[system_name]:
                    value = results[system_name][scenario]["retrieval_metrics"].get(metric, 0.0)
                    table.loc[system_name, scenario] = value
        
        tables[f"retrieval_{metric}"] = table
    
    # Classification metrics table
    classification_metrics = ["accuracy", "precision", "recall", "f1"]
    
    for metric in classification_metrics:
        table = pd.DataFrame(index=system_names, columns=scenarios)
        
        for system_name in system_names:
            for scenario in scenarios:
                if scenario in results[system_name]:
                    value = results[system_name][scenario]["classification_metrics"].get(metric, 0.0)
                    table.loc[system_name, scenario] = value
        
        tables[f"classification_{metric}"] = table
    
    # Performance metrics table
    performance_metrics = ["mean_latency", "p95_latency", "mean_memory"]
    
    for metric in performance_metrics:
        table = pd.DataFrame(index=system_names, columns=scenarios)
        
        for system_name in system_names:
            for scenario in scenarios:
                if scenario in results[system_name]:
                    value = results[system_name][scenario]["performance_metrics"].get(metric, 0.0)
                    table.loc[system_name, scenario] = value
        
        tables[f"performance_{metric}"] = table
    
    # Save tables
    for metric_name, table in tables.items():
        table_file = os.path.join(report_dir, f"{metric_name}_table.csv")
        table.to_csv(table_file)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        if "performance" in metric_name:
            # Lower is better for performance metrics
            sns.heatmap(table, annot=True, cmap="RdYlGn_r", fmt=".3f")
        else:
            # Higher is better for retrieval and classification metrics
            sns.heatmap(table, annot=True, cmap="RdYlGn", fmt=".3f")
        
        plt.title(f"{metric_name.replace('_', ' ').title()}")
        plt.tight_layout()
        
        heatmap_file = os.path.join(report_dir, f"{metric_name}_heatmap.png")
        plt.savefig(heatmap_file)
        plt.close()
    
    # Create overall ranking table
    if "comparison" in results and "overall_rankings" in results["comparison"]:
        overall_rankings = results["comparison"]["overall_rankings"]
        
        ranking_table = pd.DataFrame(index=system_names, columns=scenarios)
        
        for scenario in scenarios:
            if scenario in overall_rankings:
                for system_name in system_names:
                    if system_name in overall_rankings[scenario]:
                        ranking_table.loc[system_name, scenario] = overall_rankings[scenario][system_name]
        
        # Save ranking table
        ranking_file = os.path.join(report_dir, "overall_ranking_table.csv")
        ranking_table.to_csv(ranking_file)
        
        # Create ranking heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(ranking_table, annot=True, cmap="RdYlGn_r", fmt=".2f")
        plt.title("Overall System Rankings (Lower is Better)")
        plt.tight_layout()
        
        ranking_heatmap_file = os.path.join(report_dir, "overall_ranking_heatmap.png")
        plt.savefig(ranking_heatmap_file)
        plt.close()
    
    # Create statistical significance summary
    if "statistical_tests" in results and "tests" in results["statistical_tests"]:
        tests = results["statistical_tests"]["tests"]
        
        significance_summary = {}
        
        for scenario in scenarios:
            if scenario in tests:
                significance_summary[scenario] = {}
                
                for metric_group in tests[scenario]:
                    for metric in tests[scenario][metric_group]:
                        metric_key = f"{metric_group}_{metric}"
                        significance_summary[scenario][metric_key] = {}
                        
                        for test_key, test_result in tests[scenario][metric_group][metric].items():
                            if test_result["is_significant"]:
                                significance_summary[scenario][metric_key][test_key] = test_result["better_system"]
        
        # Save significance summary
        significance_file = os.path.join(report_dir, "statistical_significance.json")
        with open(significance_file, "w") as f:
            json.dump(significance_summary, f, indent=2)
    
    # Create summary report
    report = {
        "systems": system_names,
        "scenarios": scenarios,
        "tables": {name: table.to_dict() for name, table in tables.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save summary report
    report_file = os.path.join(report_dir, "summary_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generated summary report in {report_dir}")

def main():
    """Main function for running benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark RAG systems for network security packet analysis")
    parser.add_argument("--scenarios", nargs="+", help="Scenarios to benchmark")
    parser.add_argument("--output-dir", help="Directory to save benchmark results")
    args = parser.parse_args()
    
    # Initialize RAG systems
    systems = initialize_rag_systems()
    
    if not systems:
        logger.error("No RAG systems initialized")
        return
    
    # Load training data
    load_training_data(systems)
    
    # Run benchmark
    results = run_benchmark(
        systems=systems,
        scenarios=args.scenarios,
        output_dir=args.output_dir
    )
    
    if not results:
        logger.error("No benchmark results generated")
        return
    
    # Generate summary report
    generate_summary_report(
        results=results,
        output_dir=args.output_dir
    )
    
    logger.info("Benchmark completed successfully")

if __name__ == "__main__":
    main()
