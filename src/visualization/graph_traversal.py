import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import os

# Create directory for saving visualizations
os.makedirs('/home/ubuntu/research/visualizations/graph_traversal', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Set up matplotlib parameters for publication quality
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'  # Using a font that's likely available
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 14

# Define color scheme (colorblind-friendly)
colors = {
    'traditional': '#1f77b4',  # blue
    'graph': '#ff7f0e',        # orange
    'cache': '#2ca02c',        # green
    'hybrid': '#d62728'        # red
}

# 1. Generate synthetic graph traversal data
def generate_hop_distance_data():
    # Number of queries
    n_queries = 1000
    
    # Generate hop distances for successful retrievals
    # Most relevant information is found within 1-3 hops
    hop_distances = np.random.choice([1, 2, 3, 4, 5, 6], size=n_queries, 
                                    p=[0.15, 0.35, 0.25, 0.15, 0.07, 0.03])
    
    # Generate hop distances for different query types
    query_types = ['Simple', 'Moderate', 'Complex']
    
    query_type_distances = {}
    
    # Simple queries tend to have shorter hop distances
    query_type_distances['Simple'] = np.random.choice([1, 2, 3, 4, 5], size=n_queries//3, 
                                                     p=[0.3, 0.4, 0.2, 0.07, 0.03])
    
    # Moderate queries have medium hop distances
    query_type_distances['Moderate'] = np.random.choice([1, 2, 3, 4, 5], size=n_queries//3, 
                                                       p=[0.1, 0.3, 0.4, 0.15, 0.05])
    
    # Complex queries tend to have longer hop distances
    query_type_distances['Complex'] = np.random.choice([1, 2, 3, 4, 5, 6], size=n_queries//3, 
                                                      p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05])
    
    return {
        'hop_distances': hop_distances,
        'query_type_distances': query_type_distances,
        'query_types': query_types
    }

# 2. Generate hop count vs. performance data
def generate_hop_count_performance_data():
    # Hop count values to test
    hop_counts = list(range(1, 11))
    
    # Generate accuracy data for different hop counts
    # Accuracy increases with hop count but plateaus
    accuracy = []
    for k in hop_counts:
        base_accuracy = 0.5 + 0.4 * (1 - np.exp(-0.5 * k))  # Asymptotic to ~0.9
        noise = np.random.normal(0, 0.02)  # Add some noise
        accuracy.append(min(1.0, max(0.0, base_accuracy + noise)))  # Ensure between 0 and 1
    
    # Generate latency data for different hop counts
    # Latency increases linearly with hop count
    latency = []
    for k in hop_counts:
        base_latency = 20 + 15 * k  # Base latency plus k-dependent component
        noise = np.random.normal(0, 5)  # Add some noise
        latency.append(max(1.0, base_latency + noise))  # Ensure positive
    
    # Generate memory usage data for different hop counts
    # Memory usage increases with hop count
    memory = []
    for k in hop_counts:
        base_memory = 100 + 50 * k  # Base memory plus k-dependent component
        noise = np.random.normal(0, 10)  # Add some noise
        memory.append(max(1.0, base_memory + noise))  # Ensure positive
    
    # Generate data for different query types
    query_types = ['Simple', 'Moderate', 'Complex']
    
    query_type_accuracy = {}
    query_type_latency = {}
    
    # Simple queries have higher accuracy and lower latency
    query_type_accuracy['Simple'] = []
    query_type_latency['Simple'] = []
    for k in hop_counts:
        base_accuracy = 0.6 + 0.35 * (1 - np.exp(-0.6 * k))  # Asymptotic to ~0.95
        noise = np.random.normal(0, 0.02)
        query_type_accuracy['Simple'].append(min(1.0, max(0.0, base_accuracy + noise)))
        
        base_latency = 15 + 10 * k
        noise = np.random.normal(0, 3)
        query_type_latency['Simple'].append(max(1.0, base_latency + noise))
    
    # Moderate queries have medium accuracy and latency
    query_type_accuracy['Moderate'] = []
    query_type_latency['Moderate'] = []
    for k in hop_counts:
        base_accuracy = 0.5 + 0.4 * (1 - np.exp(-0.5 * k))  # Asymptotic to ~0.9
        noise = np.random.normal(0, 0.02)
        query_type_accuracy['Moderate'].append(min(1.0, max(0.0, base_accuracy + noise)))
        
        base_latency = 20 + 15 * k
        noise = np.random.normal(0, 5)
        query_type_latency['Moderate'].append(max(1.0, base_latency + noise))
    
    # Complex queries have lower accuracy and higher latency
    query_type_accuracy['Complex'] = []
    query_type_latency['Complex'] = []
    for k in hop_counts:
        base_accuracy = 0.4 + 0.45 * (1 - np.exp(-0.4 * k))  # Asymptotic to ~0.85
        noise = np.random.normal(0, 0.02)
        query_type_accuracy['Complex'].append(min(1.0, max(0.0, base_accuracy + noise)))
        
        base_latency = 25 + 20 * k
        noise = np.random.normal(0, 7)
        query_type_latency['Complex'].append(max(1.0, base_latency + noise))
    
    return {
        'hop_counts': hop_counts,
        'accuracy': accuracy,
        'latency': latency,
        'memory': memory,
        'query_types': query_types,
        'query_type_accuracy': query_type_accuracy,
        'query_type_latency': query_type_latency
    }

# 3. Generate TopK parameter optimization data
def generate_topk_parameter_data():
    # TopK values to test
    topk_values = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]
    
    # Generate accuracy data for different TopK values
    # Accuracy increases with TopK but plateaus
    accuracy = []
    for k in topk_values:
        base_accuracy = 0.5 + 0.4 * (1 - np.exp(-0.1 * k))  # Asymptotic to ~0.9
        noise = np.random.normal(0, 0.02)  # Add some noise
        accuracy.append(min(1.0, max(0.0, base_accuracy + noise)))  # Ensure between 0 and 1
    
    # Generate resource usage data for different TopK values
    # Resource usage increases with TopK
    resource_usage = []
    for k in topk_values:
        base_usage = 20 + 5 * k  # Base usage plus k-dependent component
        noise = np.random.normal(0, 2)  # Add some noise
        resource_usage.append(max(1.0, base_usage + noise))  # Ensure positive
    
    # Generate F1 score data for different TopK values
    # F1 score increases with TopK but plateaus and then decreases
    f1_score = []
    for k in topk_values:
        # Quadratic function that peaks around k=25
        base_f1 = 0.5 + 0.4 * (1 - ((k - 25) / 25) ** 2)
        noise = np.random.normal(0, 0.02)  # Add some noise
        f1_score.append(min(1.0, max(0.0, base_f1 + noise)))  # Ensure between 0 and 1
    
    # Generate data for different graph densities
    graph_densities = ['Sparse', 'Medium', 'Dense']
    
    density_accuracy = {}
    density_resource = {}
    
    # Sparse graphs have lower accuracy but also lower resource usage
    density_accuracy['Sparse'] = []
    density_resource['Sparse'] = []
    for k in topk_values:
        base_accuracy = 0.4 + 0.35 * (1 - np.exp(-0.1 * k))
        noise = np.random.normal(0, 0.02)
        density_accuracy['Sparse'].append(min(1.0, max(0.0, base_accuracy + noise)))
        
        base_usage = 15 + 3 * k
        noise = np.random.normal(0, 1.5)
        density_resource['Sparse'].append(max(1.0, base_usage + noise))
    
    # Medium density graphs have medium accuracy and resource usage
    density_accuracy['Medium'] = []
    density_resource['Medium'] = []
    for k in topk_values:
        base_accuracy = 0.5 + 0.4 * (1 - np.exp(-0.1 * k))
        noise = np.random.normal(0, 0.02)
        density_accuracy['Medium'].append(min(1.0, max(0.0, base_accuracy + noise)))
        
        base_usage = 20 + 5 * k
        noise = np.random.normal(0, 2)
        density_resource['Medium'].append(max(1.0, base_usage + noise))
    
    # Dense graphs have higher accuracy but also higher resource usage
    density_accuracy['Dense'] = []
    density_resource['Dense'] = []
    for k in topk_values:
        base_accuracy = 0.6 + 0.35 * (1 - np.exp(-0.1 * k))
        noise = np.random.normal(0, 0.02)
        density_accuracy['Dense'].append(min(1.0, max(0.0, base_accuracy + noise)))
        
        base_usage = 25 + 7 * k
        noise = np.random.normal(0, 2.5)
        density_resource['Dense'].append(max(1.0, base_usage + noise))
    
    return {
        'topk_values': topk_values,
        'accuracy': accuracy,
        'resource_usage': resource_usage,
        'f1_score': f1_score,
        'graph_densities': graph_densities,
        'density_accuracy': density_accuracy,
        'density_resource': density_resource
    }

# 4. Generate graph structure visualization data
def generate_graph_structure_data():
    # Create a random graph for visualization
    G = nx.random_geometric_graph(50, 0.2, seed=42)
    
    # Add node attributes
    for i, node in enumerate(G.nodes()):
        # Assign node types
        if i < 10:
            G.nodes[node]['type'] = 'ip'
        elif i < 20:
            G.nodes[node]['type'] = 'port'
        elif i < 30:
            G.nodes[node]['type'] = 'protocol'
        elif i < 40:
            G.nodes[node]['type'] = 'service'
        else:
            G.nodes[node]['type'] = 'event'
        
        # Assign relevance scores (simulated)
        G.nodes[node]['relevance'] = np.random.uniform(0, 1)
    
    # Add edge attributes
    for edge in G.edges():
        # Assign edge types
        r = np.random.random()
        if r < 0.2:
            G.edges[edge]['type'] = 'connects_to'
        elif r < 0.4:
            G.edges[edge]['type'] = 'uses'
        elif r < 0.6:
            G.edges[edge]['type'] = 'implements'
        elif r < 0.8:
            G.edges[edge]['type'] = 'provides'
        else:
            G.edges[edge]['type'] = 'generates'
        
        # Assign edge weights
        G.edges[edge]['weight'] = np.random.uniform(1, 5)
    
    # Create a subgraph for traversal visualization
    # Select a random node as the starting point
    start_node = np.random.choice(list(G.nodes()))
    
    # Create subgraphs for different hop distances
    subgraphs = {}
    for k in range(1, 4):
        # Get all nodes within k hops
        nodes = set([start_node])
        current_nodes = set([start_node])
        
        for _ in range(k):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(G.neighbors(node))
            nodes.update(next_nodes)
            current_nodes = next_nodes
        
        # Create the subgraph
        subgraphs[k] = G.subgraph(nodes)
    
    return {
        'full_graph': G,
        'start_node': start_node,
        'subgraphs': subgraphs
    }

# 1. Hop Distance Distribution Histogram
def plot_hop_distance_histogram(hop_data, filename):
    plt.figure(figsize=(14, 10))
    
    # Create histogram
    bins = np.arange(0.5, 7.5, 1)  # Bins centered on integers 1-7
    plt.hist(hop_data['hop_distances'], bins=bins, alpha=0.7, color=colors['graph'], 
             edgecolor='black', linewidth=1.2)
    
    plt.xlabel('Hop Distance')
    plt.ylabel('Number of Queries')
    plt.title('Distribution of Hop Distances for Successful Retrievals')
    
    # Set x-ticks at integer values
    plt.xticks(range(1, 8))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Query Type Hop Distance Comparison
def plot_query_type_hop_comparison(hop_data, filename):
    plt.figure(figsize=(14, 10))
    
    # Create histograms for each query type
    bins = np.arange(0.5, 7.5, 1)  # Bins centered on integers 1-7
    
    query_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    for i, query_type in enumerate(hop_data['query_types']):
        plt.hist(hop_data['query_type_distances'][query_type], bins=bins, alpha=0.7, 
                 color=query_colors[i], edgecolor='black', linewidth=1.2, 
                 label=query_type)
    
    plt.xlabel('Hop Distance')
    plt.ylabel('Number of Queries')
    plt.title('Hop Distance Distribution by Query Complexity')
    
    # Set x-ticks at integer values
    plt.xticks(range(1, 8))
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Cumulative Distribution Function
def plot_hop_distance_cdf(hop_data, filename):
    plt.figure(figsize=(14, 10))
    
    # Calculate CDF
    hop_counts = np.bincount(hop_data['hop_distances'])
    hop_counts = hop_counts / hop_counts.sum()  # Normalize
    cdf = np.cumsum(hop_counts)
    
    # Plot CDF
    plt.plot(range(len(cdf)), cdf, marker='o', linestyle='-', linewidth=2, 
             color=colors['graph'], markersize=8)
    
    plt.xlabel('Hop Distance')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function of Hop Distances')
    
    # Set x-ticks at integer values
    plt.xticks(range(len(cdf)))
    
    # Set y-ticks
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines at key percentiles
    percentiles = [0.5, 0.75, 0.9, 0.95]
    for p in percentiles:
        # Find the first hop distance where CDF >= p
        hop_at_percentile = np.where(cdf >= p)[0][0]
        
        plt.axhline(y=p, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=hop_at_percentile, color='gray', linestyle='--', alpha=0.5)
        
        plt.text(hop_at_percentile + 0.1, p - 0.03, f"{int(p*100)}% of queries: ≤{hop_at_percentile} hops", 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Hop Count vs. Accuracy
def plot_hop_count_accuracy(hop_performance_data, filename):
    plt.figure(figsize=(14, 10))
    
    # Plot accuracy vs. hop count
    plt.plot(hop_performance_data['hop_counts'], hop_performance_data['accuracy'], 
             marker='o', linestyle='-', linewidth=2, color=colors['graph'], 
             markersize=8, label='Overall Accuracy')
    
    # Add a horizontal line at 0.9 accuracy
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    
    # Find the first hop count where accuracy >= 0.9
    try:
        optimal_hop = next(i for i, acc in enumerate(hop_performance_data['accuracy']) if acc >= 0.9)
        optimal_hop = hop_performance_data['hop_counts'][optimal_hop]
        
        plt.axvline(x=optimal_hop, color='gray', linestyle='--', alpha=0.5)
        
        plt.text(optimal_hop + 0.1, 0.91, f"Optimal hop count: {optimal_hop}", 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    except StopIteration:
        # If no accuracy >= 0.9, don't add the annotation
        pass
    
    plt.xlabel('Maximum Hop Count (k)')
    plt.ylabel('Retrieval Accuracy')
    plt.title('Impact of Maximum Hop Count on Retrieval Accuracy')
    
    # Set x-ticks at integer values
    plt.xticks(hop_performance_data['hop_counts'])
    
    # Set y-range
    plt.ylim(0.4, 1.0)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Hop Count vs. Latency
def plot_hop_count_latency(hop_performance_data, filename):
    plt.figure(figsize=(14, 10))
    
    # Plot latency vs. hop count
    plt.plot(hop_performance_data['hop_counts'], hop_performance_data['latency'], 
             marker='o', linestyle='-', linewidth=2, color=colors['graph'], 
             markersize=8, label='Query Latency')
    
    plt.xlabel('Maximum Hop Count (k)')
    plt.ylabel('Query Latency (ms)')
    plt.title('Impact of Maximum Hop Count on Query Latency')
    
    # Set x-ticks at integer values
    plt.xticks(hop_performance_data['hop_counts'])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Efficiency Frontier
def plot_efficiency_frontier(hop_performance_data, filename):
    plt.figure(figsize=(14, 10))
    
    # Normalize latency (lower is better)
    max_latency = max(hop_performance_data['latency'])
    min_latency = min(hop_performance_data['latency'])
    normalized_latency = [(max_latency - l) / (max_latency - min_latency) for l in hop_performance_data['latency']]
    
    # Calculate efficiency score (weighted average of accuracy and normalized latency)
    # Higher weight on accuracy (0.7) vs. latency (0.3)
    efficiency = [0.7 * acc + 0.3 * lat for acc, lat in zip(hop_performance_data['accuracy'], normalized_latency)]
    
    # Plot efficiency vs. hop count
    plt.plot(hop_performance_data['hop_counts'], efficiency, 
             marker='o', linestyle='-', linewidth=2, color='purple', 
             markersize=8, label='Efficiency Score')
    
    # Find the hop count with maximum efficiency
    optimal_hop_idx = np.argmax(efficiency)
    optimal_hop = hop_performance_data['hop_counts'][optimal_hop_idx]
    
    plt.axvline(x=optimal_hop, color='gray', linestyle='--', alpha=0.5)
    
    plt.text(optimal_hop + 0.1, efficiency[optimal_hop_idx] - 0.03, 
             f"Optimal hop count: {optimal_hop}", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Maximum Hop Count (k)')
    plt.ylabel('Efficiency Score (0.7*Accuracy + 0.3*NormalizedSpeed)')
    plt.title('Efficiency Frontier for Hop Count Optimization')
    
    # Set x-ticks at integer values
    plt.xticks(hop_performance_data['hop_counts'])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Query Type Performance Comparison
def plot_query_type_performance(hop_performance_data, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot accuracy for each query type
    for i, query_type in enumerate(hop_performance_data['query_types']):
        ax1.plot(hop_performance_data['hop_counts'], hop_performance_data['query_type_accuracy'][query_type], 
                marker='o', linestyle='-', linewidth=2, 
                markersize=6, label=query_type)
    
    ax1.set_xlabel('Maximum Hop Count (k)')
    ax1.set_ylabel('Retrieval Accuracy')
    ax1.set_title('Impact of Hop Count on Accuracy by Query Type')
    
    # Set x-ticks at integer values
    ax1.set_xticks(hop_performance_data['hop_counts'])
    
    # Set y-range
    ax1.set_ylim(0.3, 1.0)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend()
    
    # Plot latency for each query type
    for i, query_type in enumerate(hop_performance_data['query_types']):
        ax2.plot(hop_performance_data['hop_counts'], hop_performance_data['query_type_latency'][query_type], 
                marker='o', linestyle='-', linewidth=2, 
                markersize=6, label=query_type)
    
    ax2.set_xlabel('Maximum Hop Count (k)')
    ax2.set_ylabel('Query Latency (ms)')
    ax2.set_title('Impact of Hop Count on Latency by Query Type')
    
    # Set x-ticks at integer values
    ax2.set_xticks(hop_performance_data['hop_counts'])
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 8. TopK Parameter Optimization
def plot_topk_optimization(topk_data, filename):
    plt.figure(figsize=(14, 10))
    
    # Plot accuracy and F1 score vs. TopK
    plt.plot(topk_data['topk_values'], topk_data['accuracy'], 
             marker='o', linestyle='-', linewidth=2, color=colors['graph'], 
             markersize=8, label='Accuracy')
    
    plt.plot(topk_data['topk_values'], topk_data['f1_score'], 
             marker='s', linestyle='--', linewidth=2, color='purple', 
             markersize=8, label='F1 Score')
    
    # Find the TopK with maximum F1 score
    optimal_topk_idx = np.argmax(topk_data['f1_score'])
    optimal_topk = topk_data['topk_values'][optimal_topk_idx]
    
    plt.axvline(x=optimal_topk, color='gray', linestyle='--', alpha=0.5)
    
    plt.text(optimal_topk + 1, topk_data['f1_score'][optimal_topk_idx] - 0.03, 
             f"Optimal TopK: {optimal_topk}", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('TopK Parameter Value')
    plt.ylabel('Performance Metric')
    plt.title('Impact of TopK Parameter on Retrieval Performance')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 9. TopK vs. Resource Usage
def plot_topk_resource_usage(topk_data, filename):
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Plot accuracy on primary y-axis
    ax1.plot(topk_data['topk_values'], topk_data['accuracy'], 
             marker='o', linestyle='-', linewidth=2, color=colors['graph'], 
             markersize=8, label='Accuracy')
    
    ax1.set_xlabel('TopK Parameter Value')
    ax1.set_ylabel('Accuracy', color=colors['graph'])
    ax1.tick_params(axis='y', labelcolor=colors['graph'])
    
    # Create secondary y-axis for resource usage
    ax2 = ax1.twinx()
    ax2.plot(topk_data['topk_values'], topk_data['resource_usage'], 
             marker='s', linestyle='--', linewidth=2, color='red', 
             markersize=8, label='Resource Usage')
    
    ax2.set_ylabel('Resource Usage (MB)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add title
    plt.title('Trade-off Between Accuracy and Resource Usage for TopK Parameter')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 10. Graph Density Comparison
def plot_graph_density_comparison(topk_data, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot accuracy for each graph density
    for i, density in enumerate(topk_data['graph_densities']):
        ax1.plot(topk_data['topk_values'], topk_data['density_accuracy'][density], 
                marker='o', linestyle='-', linewidth=2, 
                markersize=6, label=density)
    
    ax1.set_xlabel('TopK Parameter Value')
    ax1.set_ylabel('Retrieval Accuracy')
    ax1.set_title('Impact of TopK on Accuracy by Graph Density')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend()
    
    # Plot resource usage for each graph density
    for i, density in enumerate(topk_data['graph_densities']):
        ax2.plot(topk_data['topk_values'], topk_data['density_resource'][density], 
                marker='o', linestyle='-', linewidth=2, 
                markersize=6, label=density)
    
    ax2.set_xlabel('TopK Parameter Value')
    ax2.set_ylabel('Resource Usage (MB)')
    ax2.set_title('Impact of TopK on Resource Usage by Graph Density')
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 11. Graph Traversal Visualization
def plot_graph_traversal(graph_data, filename):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Node colors based on type
    node_colors = {
        'ip': '#1f77b4',       # blue
        'port': '#ff7f0e',     # orange
        'protocol': '#2ca02c', # green
        'service': '#d62728',  # red
        'event': '#9467bd'     # purple
    }
    
    # Edge colors based on type
    edge_colors = {
        'connects_to': '#1f77b4',  # blue
        'uses': '#ff7f0e',         # orange
        'implements': '#2ca02c',   # green
        'provides': '#d62728',     # red
        'generates': '#9467bd'     # purple
    }
    
    # Create a position layout for the full graph
    pos = nx.spring_layout(graph_data['full_graph'], seed=42)
    
    # Plot subgraphs for different hop distances
    for i, k in enumerate([1, 2, 3]):
        ax = axes[i]
        
        # Get the subgraph
        subgraph = graph_data['subgraphs'][k]
        
        # Get node colors
        node_color_list = [node_colors[subgraph.nodes[node]['type']] for node in subgraph.nodes()]
        
        # Highlight the start node
        node_size_list = []
        for node in subgraph.nodes():
            if node == graph_data['start_node']:
                node_size_list.append(300)  # Larger size for start node
            else:
                node_size_list.append(100)  # Regular size for other nodes
        
        # Get edge colors
        edge_color_list = [edge_colors[subgraph.edges[edge]['type']] for edge in subgraph.edges()]
        
        # Get edge widths
        edge_width_list = [subgraph.edges[edge]['weight'] * 0.5 for edge in subgraph.edges()]
        
        # Draw the subgraph
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_color_list, 
                              node_size=node_size_list, alpha=0.8)
        
        nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color=edge_color_list, 
                              width=edge_width_list, alpha=0.6, arrows=True, arrowsize=10)
        
        # Add labels to the start node only
        labels = {graph_data['start_node']: 'Start'}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, ax=ax, font_size=10)
        
        ax.set_title(f"{k}-Hop Neighborhood")
        ax.axis('off')
    
    plt.suptitle("Graph Traversal Visualization: k-Hop Neighborhoods", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all the data
print("Generating hop distance data...")
hop_distance_data = generate_hop_distance_data()

print("Generating hop count performance data...")
hop_count_performance_data = generate_hop_count_performance_data()

print("Generating TopK parameter data...")
topk_parameter_data = generate_topk_parameter_data()

print("Generating graph structure data...")
graph_structure_data = generate_graph_structure_data()

# Create all the visualizations
print("Creating hop distance distribution histogram...")
plot_hop_distance_histogram(hop_distance_data, '/home/ubuntu/research/visualizations/graph_traversal/hop_distance_histogram.svg')

print("Creating query type hop comparison...")
plot_query_type_hop_comparison(hop_distance_data, '/home/ubuntu/research/visualizations/graph_traversal/query_type_hop_comparison.svg')

print("Creating hop distance CDF...")
plot_hop_distance_cdf(hop_distance_data, '/home/ubuntu/research/visualizations/graph_traversal/hop_distance_cdf.svg')

print("Creating hop count vs. accuracy plot...")
plot_hop_count_accuracy(hop_count_performance_data, '/home/ubuntu/research/visualizations/graph_traversal/hop_count_accuracy.svg')

print("Creating hop count vs. latency plot...")
plot_hop_count_latency(hop_count_performance_data, '/home/ubuntu/research/visualizations/graph_traversal/hop_count_latency.svg')

print("Creating efficiency frontier plot...")
plot_efficiency_frontier(hop_count_performance_data, '/home/ubuntu/research/visualizations/graph_traversal/efficiency_frontier.svg')

print("Creating query type performance comparison...")
plot_query_type_performance(hop_count_performance_data, '/home/ubuntu/research/visualizations/graph_traversal/query_type_performance.svg')

print("Creating TopK optimization plot...")
plot_topk_optimization(topk_parameter_data, '/home/ubuntu/research/visualizations/graph_traversal/topk_optimization.svg')

print("Creating TopK vs. resource usage plot...")
plot_topk_resource_usage(topk_parameter_data, '/home/ubuntu/research/visualizations/graph_traversal/topk_resource_usage.svg')

print("Creating graph density comparison...")
plot_graph_density_comparison(topk_parameter_data, '/home/ubuntu/research/visualizations/graph_traversal/graph_density_comparison.svg')

print("Creating graph traversal visualization...")
plot_graph_traversal(graph_structure_data, '/home/ubuntu/research/visualizations/graph_traversal/graph_traversal_visualization.svg')

print("Graph traversal analytics visualizations generated successfully.")
