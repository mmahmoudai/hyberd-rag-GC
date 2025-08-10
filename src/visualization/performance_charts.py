import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os

# Create directory for saving visualizations
os.makedirs('/home/ubuntu/research/visualizations/performance_charts', exist_ok=True)

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

# Generate synthetic data for performance comparison
def generate_synthetic_data():
    # Number of samples
    n_samples = 1000
    
    # True labels (1 for attack, 0 for normal)
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Generate prediction scores for each RAG approach
    # Traditional RAG - decent performance
    traditional_scores = np.random.beta(8, 4, n_samples) * y_true + np.random.beta(4, 8, n_samples) * (1 - y_true)
    
    # Graph RAG - better performance
    graph_scores = np.random.beta(10, 3, n_samples) * y_true + np.random.beta(3, 10, n_samples) * (1 - y_true)
    
    # Cache RAG - similar to traditional but slightly better
    cache_scores = np.random.beta(9, 4, n_samples) * y_true + np.random.beta(4, 9, n_samples) * (1 - y_true)
    
    # Hybrid RAG - best performance
    hybrid_scores = np.random.beta(12, 2, n_samples) * y_true + np.random.beta(2, 12, n_samples) * (1 - y_true)
    
    # Create a dictionary with all data
    data = {
        'y_true': y_true,
        'traditional_scores': traditional_scores,
        'graph_scores': graph_scores,
        'cache_scores': cache_scores,
        'hybrid_scores': hybrid_scores
    }
    
    return data

# Generate data for different attack types
def generate_attack_type_data():
    attack_types = ['DDoS', 'Port Scan', 'Brute Force', 'Data Exfiltration']
    all_data = {}
    
    for attack in attack_types:
        # Adjust parameters to simulate different detection difficulty for different attacks
        if attack == 'DDoS':
            # DDoS is easier to detect
            traditional_params = (9, 3, 3, 9)
            graph_params = (11, 2, 2, 11)
            cache_params = (10, 3, 3, 10)
            hybrid_params = (13, 1, 1, 13)
        elif attack == 'Port Scan':
            # Port Scan is moderately difficult
            traditional_params = (8, 4, 4, 8)
            graph_params = (10, 3, 3, 10)
            cache_params = (9, 4, 4, 9)
            hybrid_params = (12, 2, 2, 12)
        elif attack == 'Brute Force':
            # Brute Force is also moderately difficult
            traditional_params = (7, 5, 5, 7)
            graph_params = (9, 4, 4, 9)
            cache_params = (8, 5, 5, 8)
            hybrid_params = (11, 3, 3, 11)
        else:  # Data Exfiltration
            # Data Exfiltration is hardest to detect
            traditional_params = (6, 6, 6, 6)
            graph_params = (8, 5, 5, 8)
            cache_params = (7, 6, 6, 7)
            hybrid_params = (10, 4, 4, 10)
        
        n_samples = 500
        y_true = np.random.binomial(1, 0.3, n_samples)
        
        traditional_scores = np.random.beta(traditional_params[0], traditional_params[1], n_samples) * y_true + \
                            np.random.beta(traditional_params[2], traditional_params[3], n_samples) * (1 - y_true)
        
        graph_scores = np.random.beta(graph_params[0], graph_params[1], n_samples) * y_true + \
                      np.random.beta(graph_params[2], graph_params[3], n_samples) * (1 - y_true)
        
        cache_scores = np.random.beta(cache_params[0], cache_params[1], n_samples) * y_true + \
                      np.random.beta(cache_params[2], cache_params[3], n_samples) * (1 - y_true)
        
        hybrid_scores = np.random.beta(hybrid_params[0], hybrid_params[1], n_samples) * y_true + \
                       np.random.beta(hybrid_params[2], hybrid_params[3], n_samples) * (1 - y_true)
        
        all_data[attack] = {
            'y_true': y_true,
            'traditional_scores': traditional_scores,
            'graph_scores': graph_scores,
            'cache_scores': cache_scores,
            'hybrid_scores': hybrid_scores
        }
    
    return all_data, attack_types

# Generate latency and resource utilization data
def generate_performance_data():
    # Number of measurements
    n_measurements = 100
    
    # Scenarios
    scenarios = ['Standard', 'Zero-Day', 'High-Throughput', 'Adversarial']
    
    # Create a dictionary to store all performance data
    performance_data = {
        'latency': {},
        'cpu_usage': {},
        'memory_usage': {},
        'network_overhead': {}
    }
    
    # Generate latency data (in milliseconds)
    # Traditional RAG - moderate latency
    traditional_latency = {
        'Standard': np.random.gamma(shape=7, scale=50, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=7.5, scale=50, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=10, scale=70, size=n_measurements),
        'Adversarial': np.random.gamma(shape=8, scale=50, size=n_measurements)
    }
    
    # Graph RAG - highest latency
    graph_latency = {
        'Standard': np.random.gamma(shape=8, scale=55, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=8.5, scale=55, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=12, scale=75, size=n_measurements),
        'Adversarial': np.random.gamma(shape=9, scale=55, size=n_measurements)
    }
    
    # Cache RAG - lowest latency
    cache_latency = {
        'Standard': np.random.gamma(shape=5, scale=45, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=5.5, scale=45, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=7, scale=50, size=n_measurements),
        'Adversarial': np.random.gamma(shape=6, scale=45, size=n_measurements)
    }
    
    # Hybrid RAG - moderate to low latency
    hybrid_latency = {
        'Standard': np.random.gamma(shape=6, scale=48, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=6.5, scale=48, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=8, scale=60, size=n_measurements),
        'Adversarial': np.random.gamma(shape=7, scale=48, size=n_measurements)
    }
    
    performance_data['latency'] = {
        'traditional': traditional_latency,
        'graph': graph_latency,
        'cache': cache_latency,
        'hybrid': hybrid_latency
    }
    
    # Generate CPU usage data (percentage)
    # Traditional RAG - moderate CPU usage
    traditional_cpu = {
        'Standard': np.random.beta(4, 8, n_measurements) * 100,
        'Zero-Day': np.random.beta(4.5, 8, n_measurements) * 100,
        'High-Throughput': np.random.beta(7, 5, n_measurements) * 100,
        'Adversarial': np.random.beta(5, 8, n_measurements) * 100
    }
    
    # Graph RAG - highest CPU usage
    graph_cpu = {
        'Standard': np.random.beta(5, 7, n_measurements) * 100,
        'Zero-Day': np.random.beta(5.5, 7, n_measurements) * 100,
        'High-Throughput': np.random.beta(8, 4, n_measurements) * 100,
        'Adversarial': np.random.beta(6, 7, n_measurements) * 100
    }
    
    # Cache RAG - lowest CPU usage
    cache_cpu = {
        'Standard': np.random.beta(3, 9, n_measurements) * 100,
        'Zero-Day': np.random.beta(3.5, 9, n_measurements) * 100,
        'High-Throughput': np.random.beta(6, 6, n_measurements) * 100,
        'Adversarial': np.random.beta(4, 9, n_measurements) * 100
    }
    
    # Hybrid RAG - moderate to high CPU usage
    hybrid_cpu = {
        'Standard': np.random.beta(4.5, 7.5, n_measurements) * 100,
        'Zero-Day': np.random.beta(5, 7.5, n_measurements) * 100,
        'High-Throughput': np.random.beta(7.5, 4.5, n_measurements) * 100,
        'Adversarial': np.random.beta(5.5, 7.5, n_measurements) * 100
    }
    
    performance_data['cpu_usage'] = {
        'traditional': traditional_cpu,
        'graph': graph_cpu,
        'cache': cache_cpu,
        'hybrid': hybrid_cpu
    }
    
    # Generate memory usage data (MB)
    # Traditional RAG - moderate memory usage
    traditional_memory = {
        'Standard': np.random.gamma(shape=10, scale=25, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=10.5, scale=25, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=15, scale=35, size=n_measurements),
        'Adversarial': np.random.gamma(shape=11, scale=25, size=n_measurements)
    }
    
    # Graph RAG - highest memory usage
    graph_memory = {
        'Standard': np.random.gamma(shape=12, scale=27, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=12.5, scale=27, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=18, scale=37, size=n_measurements),
        'Adversarial': np.random.gamma(shape=13, scale=27, size=n_measurements)
    }
    
    # Cache RAG - high memory usage
    cache_memory = {
        'Standard': np.random.gamma(shape=11, scale=26, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=11.5, scale=26, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=16, scale=27, size=n_measurements),
        'Adversarial': np.random.gamma(shape=12, scale=26, size=n_measurements)
    }
    
    # Hybrid RAG - highest memory usage
    hybrid_memory = {
        'Standard': np.random.gamma(shape=13, scale=28, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=13.5, scale=28, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=19, scale=32, size=n_measurements),
        'Adversarial': np.random.gamma(shape=14, scale=28, size=n_measurements)
    }
    
    performance_data['memory_usage'] = {
        'traditional': traditional_memory,
        'graph': graph_memory,
        'cache': cache_memory,
        'hybrid': hybrid_memory
    }
    
    # Generate network overhead data (MB/s)
    # Traditional RAG - moderate network overhead
    traditional_network = {
        'Standard': np.random.gamma(shape=3, scale=1.7, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=3.2, scale=1.7, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=5, scale=2.5, size=n_measurements),
        'Adversarial': np.random.gamma(shape=3.5, scale=1.7, size=n_measurements)
    }
    
    # Graph RAG - highest network overhead
    graph_network = {
        'Standard': np.random.gamma(shape=4, scale=1.8, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=4.2, scale=1.8, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=6, scale=2.6, size=n_measurements),
        'Adversarial': np.random.gamma(shape=4.5, scale=1.8, size=n_measurements)
    }
    
    # Cache RAG - lowest network overhead
    cache_network = {
        'Standard': np.random.gamma(shape=2, scale=1.6, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=2.2, scale=1.6, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=4, scale=2.0, size=n_measurements),
        'Adversarial': np.random.gamma(shape=2.5, scale=1.6, size=n_measurements)
    }
    
    # Hybrid RAG - moderate network overhead
    hybrid_network = {
        'Standard': np.random.gamma(shape=3, scale=1.7, size=n_measurements),
        'Zero-Day': np.random.gamma(shape=3.2, scale=1.7, size=n_measurements),
        'High-Throughput': np.random.gamma(shape=5, scale=2.4, size=n_measurements),
        'Adversarial': np.random.gamma(shape=3.5, scale=1.7, size=n_measurements)
    }
    
    performance_data['network_overhead'] = {
        'traditional': traditional_network,
        'graph': graph_network,
        'cache': cache_network,
        'hybrid': hybrid_network
    }
    
    return performance_data, scenarios

# 1. ROC Curves
def plot_roc_curves(data, filename):
    plt.figure(figsize=(12, 10))
    
    # Calculate ROC curve and AUC for each approach
    approaches = ['traditional', 'graph', 'cache', 'hybrid']
    approach_names = ['Traditional RAG', 'Graph RAG', 'Cache RAG', 'Hybrid Cache-Graph RAG']
    
    for i, approach in enumerate(approaches):
        scores = data[f'{approach}_scores']
        fpr, tpr, _ = roc_curve(data['y_true'], scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, color=colors[approach],
                 label=f'{approach_names[i]} (AUC = {roc_auc:.3f})')
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different RAG Approaches in Network Security')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Precision-Recall Curves
def plot_pr_curves(data, filename):
    plt.figure(figsize=(12, 10))
    
    # Calculate Precision-Recall curve for each approach
    approaches = ['traditional', 'graph', 'cache', 'hybrid']
    approach_names = ['Traditional RAG', 'Graph RAG', 'Cache RAG', 'Hybrid Cache-Graph RAG']
    
    for i, approach in enumerate(approaches):
        scores = data[f'{approach}_scores']
        precision, recall, _ = precision_recall_curve(data['y_true'], scores)
        avg_precision = np.mean(precision)
        
        plt.plot(recall, precision, lw=2, color=colors[approach],
                 label=f'{approach_names[i]} (Avg. Precision = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Different RAG Approaches in Network Security')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Attack-Specific ROC Curves
def plot_attack_specific_roc(attack_data, attack_types, filename):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    approaches = ['traditional', 'graph', 'cache', 'hybrid']
    approach_names = ['Traditional RAG', 'Graph RAG', 'Cache RAG', 'Hybrid Cache-Graph RAG']
    
    for i, attack in enumerate(attack_types):
        ax = axes[i]
        
        for j, approach in enumerate(approaches):
            scores = attack_data[attack][f'{approach}_scores']
            fpr, tpr, _ = roc_curve(attack_data[attack]['y_true'], scores)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, color=colors[approach],
                   label=f'{approach_names[j]} (AUC = {roc_auc:.3f})')
        
        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve: {attack} Attack Detection')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Latency Comparison Bar Chart
def plot_latency_comparison(performance_data, scenarios, filename):
    plt.figure(figsize=(14, 10))
    
    # Calculate mean latency for each approach and scenario
    approaches = ['traditional', 'graph', 'cache', 'hybrid']
    approach_names = ['Traditional RAG', 'Graph RAG', 'Cache RAG', 'Hybrid Cache-Graph RAG']
    
    # Prepare data for plotting
    mean_latencies = {}
    for approach in approaches:
        mean_latencies[approach] = [np.mean(performance_data['latency'][approach][scenario]) for scenario in scenarios]
    
    # Set up bar positions
    x = np.arange(len(scenarios))
    width = 0.2  # Width of the bars
    
    # Plot bars for each approach
    for i, approach in enumerate(approaches):
        plt.bar(x + (i - 1.5) * width, mean_latencies[approach], width, 
                color=colors[approach], label=approach_names[i])
    
    plt.xlabel('Scenario')
    plt.ylabel('Mean Latency (ms)')
    plt.title('Latency Comparison Across Different RAG Approaches and Scenarios')
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, approach in enumerate(approaches):
        for j, v in enumerate(mean_latencies[approach]):
            plt.text(j + (i - 1.5) * width, v + 5, f'{v:.1f}', 
                     ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Latency Distribution Box Plots
def plot_latency_boxplots(performance_data, scenarios, filename):
    plt.figure(figsize=(16, 10))
    
    approaches = ['traditional', 'graph', 'cache', 'hybrid']
    approach_names = ['Traditional RAG', 'Graph RAG', 'Cache RAG', 'Hybrid Cache-Graph RAG']
    
    # Prepare data for box plots
    data_for_boxplot = []
    labels = []
    
    for scenario in scenarios:
        for i, approach in enumerate(approaches):
            data_for_boxplot.append(performance_data['latency'][approach][scenario])
            labels.append(f"{approach_names[i]}\n({scenario})")
    
    # Create box plots
    box_colors = []
    for scenario in scenarios:
        for approach in approaches:
            box_colors.append(colors[approach])
    
    # Create the box plot
    bp = plt.boxplot(data_for_boxplot, patch_artist=True, labels=labels)
    
    # Set box colors
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=box_colors[i], alpha=0.7)
        box.set(edgecolor='black')
    
    # Set whisker and cap colors
    for i, whisker in enumerate(bp['whiskers']):
        whisker.set(color='black')
    
    for i, cap in enumerate(bp['caps']):
        cap.set(color='black')
    
    # Set median line colors
    for i, median in enumerate(bp['medians']):
        median.set(color='black')
    
    plt.ylabel('Latency (ms)')
    plt.title('Latency Distribution Across Different RAG Approaches and Scenarios')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Create a custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[approach], edgecolor='black', label=name)
                      for approach, name in zip(approaches, approach_names)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Resource Utilization Line Graphs
def plot_resource_utilization(performance_data, filename_prefix):
    approaches = ['traditional', 'graph', 'cache', 'hybrid']
    approach_names = ['Traditional RAG', 'Graph RAG', 'Cache RAG', 'Hybrid Cache-Graph RAG']
    
    # Resource metrics to plot
    resources = {
        'cpu_usage': {'title': 'CPU Usage Over Time', 'ylabel': 'CPU Usage (%)', 'filename': 'cpu_usage.svg'},
        'memory_usage': {'title': 'Memory Usage Over Time', 'ylabel': 'Memory Usage (MB)', 'filename': 'memory_usage.svg'},
        'network_overhead': {'title': 'Network Overhead Over Time', 'ylabel': 'Network Overhead (MB/s)', 'filename': 'network_overhead.svg'}
    }
    
    # Plot each resource metric
    for resource, config in resources.items():
        plt.figure(figsize=(14, 10))
        
        # Use the 'Standard' scenario for the time series
        x = np.arange(len(performance_data[resource]['traditional']['Standard']))
        
        for i, approach in enumerate(approaches):
            plt.plot(x, performance_data[resource][approach]['Standard'], 
                    color=colors[approach], label=approach_names[i], linewidth=2)
        
        plt.xlabel('Time (s)')
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(f"{filename_prefix}{config['filename']}", format='svg', dpi=300, bbox_inches='tight')
        plt.close()

# 7. Resource Efficiency Radar Chart
def plot_resource_efficiency_radar(performance_data, scenarios, filename):
    # We'll use the 'Standard' scenario for the radar chart
    scenario = 'Standard'
    
    approaches = ['traditional', 'graph', 'cache', 'hybrid']
    approach_names = ['Traditional RAG', 'Graph RAG', 'Cache RAG', 'Hybrid Cache-Graph RAG']
    
    # Calculate mean values for each metric
    metrics = ['latency', 'cpu_usage', 'memory_usage', 'network_overhead']
    metric_names = ['Latency', 'CPU Usage', 'Memory Usage', 'Network Overhead']
    
    # Normalize values for radar chart (lower is better for all metrics)
    normalized_values = {}
    
    for metric in metrics:
        max_val = max([np.mean(performance_data[metric][approach][scenario]) for approach in approaches])
        min_val = min([np.mean(performance_data[metric][approach][scenario]) for approach in approaches])
        
        normalized_values[metric] = {}
        for approach in approaches:
            # Invert the normalization since lower values are better
            val = np.mean(performance_data[metric][approach][scenario])
            if max_val > min_val:  # Avoid division by zero
                normalized_values[metric][approach] = 1 - ((val - min_val) / (max_val - min_val))
            else:
                normalized_values[metric][approach] = 1.0
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    for i, approach in enumerate(approaches):
        values = [normalized_values[metric][approach] for metric in metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, color=colors[approach], linewidth=2, label=approach_names[i])
        ax.fill(angles, values, color=colors[approach], alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Resource Efficiency Comparison (Higher is Better)', size=15)
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all the data
print("Generating synthetic data for performance visualizations...")
classification_data = generate_synthetic_data()
attack_data, attack_types = generate_attack_type_data()
performance_data, scenarios = generate_performance_data()

# Create all the visualizations
print("Creating ROC curves...")
plot_roc_curves(classification_data, '/home/ubuntu/research/visualizations/performance_charts/roc_curves.svg')

print("Creating Precision-Recall curves...")
plot_pr_curves(classification_data, '/home/ubuntu/research/visualizations/performance_charts/pr_curves.svg')

print("Creating attack-specific ROC curves...")
plot_attack_specific_roc(attack_data, attack_types, '/home/ubuntu/research/visualizations/performance_charts/attack_specific_roc.svg')

print("Creating latency comparison bar chart...")
plot_latency_comparison(performance_data, scenarios, '/home/ubuntu/research/visualizations/performance_charts/latency_comparison.svg')

print("Creating latency distribution box plots...")
plot_latency_boxplots(performance_data, scenarios, '/home/ubuntu/research/visualizations/performance_charts/latency_boxplots.svg')

print("Creating resource utilization line graphs...")
plot_resource_utilization(performance_data, '/home/ubuntu/research/visualizations/performance_charts/')

print("Creating resource efficiency radar chart...")
plot_resource_efficiency_radar(performance_data, scenarios, '/home/ubuntu/research/visualizations/performance_charts/resource_efficiency_radar.svg')

print("Performance comparison charts generated successfully.")
