import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.dates import DateFormatter, MinuteLocator
import datetime

# Create directory for saving visualizations
os.makedirs('/home/ubuntu/research/visualizations/cache_performance', exist_ok=True)

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

# 1. Cache Hit Rate Heatmaps
def generate_cache_hit_rate_data():
    # Cache sizes in MB
    cache_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    
    # Query complexity levels
    complexity_levels = ['Simple', 'Moderate', 'Complex', 'Very Complex']
    
    # Network traffic volumes (packets per second)
    traffic_volumes = [1000, 5000, 10000, 50000, 100000]
    
    # Generate hit rate data for cache sizes vs. query complexity
    hit_rates_complexity = np.zeros((len(cache_sizes), len(complexity_levels)))
    
    # Fill with realistic hit rate data
    # Larger caches have higher hit rates
    # More complex queries have lower hit rates
    for i, size in enumerate(cache_sizes):
        base_hit_rate = 0.5 + 0.4 * (i / (len(cache_sizes) - 1))  # 0.5 to 0.9
        for j, complexity in enumerate(complexity_levels):
            complexity_factor = 1 - 0.2 * (j / (len(complexity_levels) - 1))  # 1.0 to 0.8
            hit_rates_complexity[i, j] = base_hit_rate * complexity_factor
            
            # Add some noise
            hit_rates_complexity[i, j] += np.random.normal(0, 0.03)
            
            # Ensure hit rates are between 0 and 1
            hit_rates_complexity[i, j] = max(0, min(1, hit_rates_complexity[i, j]))
    
    # Generate hit rate data for cache sizes vs. traffic volumes
    hit_rates_traffic = np.zeros((len(cache_sizes), len(traffic_volumes)))
    
    # Fill with realistic hit rate data
    # Larger caches have higher hit rates
    # Higher traffic volumes have lower hit rates due to more diverse data
    for i, size in enumerate(cache_sizes):
        base_hit_rate = 0.5 + 0.4 * (i / (len(cache_sizes) - 1))  # 0.5 to 0.9
        for j, volume in enumerate(traffic_volumes):
            volume_factor = 1 - 0.3 * (j / (len(traffic_volumes) - 1))  # 1.0 to 0.7
            hit_rates_traffic[i, j] = base_hit_rate * volume_factor
            
            # Add some noise
            hit_rates_traffic[i, j] += np.random.normal(0, 0.03)
            
            # Ensure hit rates are between 0 and 1
            hit_rates_traffic[i, j] = max(0, min(1, hit_rates_traffic[i, j]))
    
    # Generate hit rate data for query complexity vs. traffic volumes
    hit_rates_complexity_traffic = np.zeros((len(complexity_levels), len(traffic_volumes)))
    
    # Fill with realistic hit rate data
    # Simpler queries have higher hit rates
    # Lower traffic volumes have higher hit rates
    for i, complexity in enumerate(complexity_levels):
        base_hit_rate = 0.9 - 0.4 * (i / (len(complexity_levels) - 1))  # 0.9 to 0.5
        for j, volume in enumerate(traffic_volumes):
            volume_factor = 1 - 0.3 * (j / (len(traffic_volumes) - 1))  # 1.0 to 0.7
            hit_rates_complexity_traffic[i, j] = base_hit_rate * volume_factor
            
            # Add some noise
            hit_rates_complexity_traffic[i, j] += np.random.normal(0, 0.03)
            
            # Ensure hit rates are between 0 and 1
            hit_rates_complexity_traffic[i, j] = max(0, min(1, hit_rates_complexity_traffic[i, j]))
    
    return {
        'cache_sizes': cache_sizes,
        'complexity_levels': complexity_levels,
        'traffic_volumes': traffic_volumes,
        'hit_rates_complexity': hit_rates_complexity,
        'hit_rates_traffic': hit_rates_traffic,
        'hit_rates_complexity_traffic': hit_rates_complexity_traffic
    }

def plot_cache_hit_rate_heatmaps(cache_data, filename_prefix):
    # 1. Cache Size vs. Query Complexity Heatmap
    plt.figure(figsize=(14, 10))
    
    # Format cache sizes for display
    cache_size_labels = [f"{size/1024:.1f} GB" if size >= 1024 else f"{size} MB" 
                         for size in cache_data['cache_sizes']]
    
    # Create heatmap
    ax = sns.heatmap(cache_data['hit_rates_complexity'], 
                    annot=True, 
                    fmt=".2f", 
                    cmap="YlGnBu",
                    xticklabels=cache_data['complexity_levels'],
                    yticklabels=cache_size_labels,
                    cbar_kws={'label': 'Hit Rate'})
    
    plt.xlabel('Query Complexity')
    plt.ylabel('Cache Size')
    plt.title('Cache Hit Rate by Cache Size and Query Complexity')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}hit_rate_size_complexity.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cache Size vs. Traffic Volume Heatmap
    plt.figure(figsize=(14, 10))
    
    # Format traffic volumes for display
    traffic_volume_labels = [f"{volume/1000:.0f}K pps" if volume >= 1000 else f"{volume} pps" 
                            for volume in cache_data['traffic_volumes']]
    
    # Create heatmap
    ax = sns.heatmap(cache_data['hit_rates_traffic'], 
                    annot=True, 
                    fmt=".2f", 
                    cmap="YlGnBu",
                    xticklabels=traffic_volume_labels,
                    yticklabels=cache_size_labels,
                    cbar_kws={'label': 'Hit Rate'})
    
    plt.xlabel('Traffic Volume (packets per second)')
    plt.ylabel('Cache Size')
    plt.title('Cache Hit Rate by Cache Size and Traffic Volume')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}hit_rate_size_traffic.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Query Complexity vs. Traffic Volume Heatmap
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    ax = sns.heatmap(cache_data['hit_rates_complexity_traffic'], 
                    annot=True, 
                    fmt=".2f", 
                    cmap="YlGnBu",
                    xticklabels=traffic_volume_labels,
                    yticklabels=cache_data['complexity_levels'],
                    cbar_kws={'label': 'Hit Rate'})
    
    plt.xlabel('Traffic Volume (packets per second)')
    plt.ylabel('Query Complexity')
    plt.title('Cache Hit Rate by Query Complexity and Traffic Volume')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}hit_rate_complexity_traffic.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Cache Invalidation Timing Diagrams
def generate_cache_invalidation_data():
    # Time period (minutes)
    time_period = 60
    time_points = np.arange(0, time_period, 0.1)
    
    # Generate graph change events (random times)
    np.random.seed(42)
    graph_change_times = np.sort(np.random.choice(np.arange(5, time_period-5), size=8, replace=False))
    
    # Generate cache invalidation events
    # Some invalidations happen right after graph changes, others are independent
    invalidation_times = []
    
    # Add invalidations triggered by graph changes (with small delay)
    for change_time in graph_change_times:
        delay = np.random.uniform(0.2, 1.0)
        invalidation_times.append(change_time + delay)
    
    # Add some independent invalidations
    independent_invalidations = np.sort(np.random.choice(np.arange(5, time_period-5), size=5, replace=False))
    invalidation_times.extend(independent_invalidations)
    invalidation_times = np.sort(invalidation_times)
    
    # Generate cache hit rate over time
    hit_rate = np.ones(len(time_points)) * 0.85  # Start with 85% hit rate
    
    # Decrease hit rate after each invalidation, then gradually recover
    for inv_time in invalidation_times:
        # Find index in time_points closest to invalidation time
        idx = np.abs(time_points - inv_time).argmin()
        
        # Decrease hit rate at invalidation
        decrease = np.random.uniform(0.15, 0.4)  # Random decrease between 15-40%
        hit_rate[idx:] -= decrease
        
        # Gradual recovery over next few minutes
        recovery_period = np.random.uniform(3, 8)  # Recovery period in minutes
        recovery_end_idx = np.abs(time_points - (inv_time + recovery_period)).argmin()
        
        if recovery_end_idx > idx:
            # Linear recovery
            hit_rate[idx:recovery_end_idx] = np.linspace(hit_rate[idx], hit_rate[idx] + decrease, recovery_end_idx - idx)
            hit_rate[recovery_end_idx:] += decrease
    
    # Ensure hit rate is between 0 and 1
    hit_rate = np.clip(hit_rate, 0, 1)
    
    # Generate efficiency impact data
    # Efficiency is measured as queries per second
    efficiency = np.ones(len(time_points)) * 100  # Start with 100 QPS
    
    # Decrease efficiency after each invalidation, then gradually recover
    for inv_time in invalidation_times:
        # Find index in time_points closest to invalidation time
        idx = np.abs(time_points - inv_time).argmin()
        
        # Decrease efficiency at invalidation
        decrease = np.random.uniform(20, 50)  # Random decrease between 20-50 QPS
        efficiency[idx:] -= decrease
        
        # Gradual recovery over next few minutes
        recovery_period = np.random.uniform(2, 5)  # Recovery period in minutes
        recovery_end_idx = np.abs(time_points - (inv_time + recovery_period)).argmin()
        
        if recovery_end_idx > idx:
            # Linear recovery
            efficiency[idx:recovery_end_idx] = np.linspace(efficiency[idx], efficiency[idx] + decrease, recovery_end_idx - idx)
            efficiency[recovery_end_idx:] += decrease
    
    # Ensure efficiency is positive
    efficiency = np.maximum(efficiency, 10)
    
    return {
        'time_points': time_points,
        'graph_change_times': graph_change_times,
        'invalidation_times': invalidation_times,
        'hit_rate': hit_rate,
        'efficiency': efficiency
    }

def plot_cache_invalidation_diagrams(invalidation_data, filename_prefix):
    # 1. Timeline visualization of invalidation events
    plt.figure(figsize=(16, 8))
    
    # Convert time points to datetime for better formatting
    start_time = datetime.datetime(2025, 4, 21, 10, 0, 0)  # Start at 10:00 AM
    
    # Convert numpy values to Python native types to avoid type errors
    time_points = [float(t) for t in invalidation_data['time_points']]
    graph_change_times = [float(t) for t in invalidation_data['graph_change_times']]
    invalidation_times = [float(t) for t in invalidation_data['invalidation_times']]
    
    time_datetimes = [start_time + datetime.timedelta(minutes=t) for t in time_points]
    graph_change_datetimes = [start_time + datetime.timedelta(minutes=t) for t in graph_change_times]
    invalidation_datetimes = [start_time + datetime.timedelta(minutes=t) for t in invalidation_times]
    
    # Plot hit rate over time
    plt.plot(time_datetimes, invalidation_data['hit_rate'], color='blue', linewidth=2, label='Cache Hit Rate')
    
    # Mark graph change events
    for change_time in graph_change_datetimes:
        plt.axvline(x=change_time, color='orange', linestyle='--', alpha=0.7)
    
    # Mark cache invalidation events
    for inv_time in invalidation_datetimes:
        plt.axvline(x=inv_time, color='red', linestyle='-', alpha=0.5)
    
    # Add legend with custom elements
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='orange', linestyle='--', lw=2),
        Line2D([0], [0], color='red', linestyle='-', lw=2, alpha=0.5)
    ]
    plt.legend(custom_lines, ['Cache Hit Rate', 'Graph Topology Change', 'Cache Invalidation'], loc='lower right')
    
    plt.xlabel('Time')
    plt.ylabel('Cache Hit Rate')
    plt.title('Cache Invalidation Timeline and Hit Rate Impact')
    
    # Format x-axis to show time
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(MinuteLocator(interval=5))
    plt.gcf().autofmt_xdate()
    
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}invalidation_timeline.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation between graph changes and cache invalidations
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot of graph changes vs invalidations
    # X-axis: time of graph change
    # Y-axis: delay until next invalidation
    
    delays = []
    change_times_minutes = []
    
    for change_time in graph_change_times:
        # Find the next invalidation after this change
        next_invalidations = [t for t in invalidation_times if t > change_time]
        
        if len(next_invalidations) > 0:
            delay = next_invalidations[0] - change_time
            
            # Only include if delay is reasonably small (likely related)
            if delay < 3:  # Less than 3 minutes
                delays.append(delay)
                change_times_minutes.append(change_time)
    
    # Convert to numpy arrays
    delays = np.array(delays)
    change_times_minutes = np.array(change_times_minutes)
    
    # Create scatter plot
    plt.scatter(change_times_minutes, delays, s=100, color='purple', alpha=0.7)
    
    # Add trend line
    if len(delays) > 1:
        z = np.polyfit(change_times_minutes, delays, 1)
        p = np.poly1d(z)
        plt.plot(change_times_minutes, p(change_times_minutes), "r--", alpha=0.7)
    
    plt.xlabel('Time of Graph Change (minutes)')
    plt.ylabel('Delay Until Cache Invalidation (minutes)')
    plt.title('Correlation Between Graph Changes and Cache Invalidations')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}invalidation_correlation.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency impact visualization
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot hit rate on primary y-axis
    ax1.plot(time_datetimes, invalidation_data['hit_rate'], color='blue', linewidth=2, label='Cache Hit Rate')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cache Hit Rate', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.05)
    
    # Create secondary y-axis for efficiency
    ax2 = ax1.twinx()
    ax2.plot(time_datetimes, invalidation_data['efficiency'], color='green', linewidth=2, label='Query Throughput')
    ax2.set_ylabel('Query Throughput (queries/second)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Mark cache invalidation events
    for inv_time in invalidation_datetimes:
        plt.axvline(x=inv_time, color='red', linestyle='-', alpha=0.3)
    
    # Add legend with custom elements
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='green', lw=2),
        Line2D([0], [0], color='red', linestyle='-', lw=2, alpha=0.3)
    ]
    plt.legend(custom_lines, ['Cache Hit Rate', 'Query Throughput', 'Cache Invalidation'], loc='lower right')
    
    plt.title('Impact of Cache Invalidations on System Efficiency')
    
    # Format x-axis to show time
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(MinuteLocator(interval=5))
    fig.autofmt_xdate()
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}invalidation_efficiency_impact.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Time-series Performance Plots
def generate_cache_performance_time_series():
    # Time period (hours)
    time_period = 24
    time_points = np.arange(0, time_period, 0.1)
    
    # Generate hit rate data for different cache strategies
    # LRU (Least Recently Used)
    lru_hit_rate = 0.75 + 0.1 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    lru_hit_rate -= 0.1 * (time_points / time_period)
    # Add some noise
    lru_hit_rate += np.random.normal(0, 0.02, size=len(time_points))
    
    # LFU (Least Frequently Used)
    lfu_hit_rate = 0.78 + 0.1 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    lfu_hit_rate -= 0.08 * (time_points / time_period)
    # Add some noise
    lfu_hit_rate += np.random.normal(0, 0.02, size=len(time_points))
    
    # ARC (Adaptive Replacement Cache)
    arc_hit_rate = 0.82 + 0.08 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    arc_hit_rate -= 0.05 * (time_points / time_period)
    # Add some noise
    arc_hit_rate += np.random.normal(0, 0.02, size=len(time_points))
    
    # Security-Aware TTL
    sa_ttl_hit_rate = 0.80 + 0.09 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    sa_ttl_hit_rate -= 0.03 * (time_points / time_period)
    # Add some noise
    sa_ttl_hit_rate += np.random.normal(0, 0.02, size=len(time_points))
    
    # Hybrid Cache-Graph RAG
    hybrid_hit_rate = 0.85 + 0.07 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    hybrid_hit_rate -= 0.02 * (time_points / time_period)
    # Add some noise
    hybrid_hit_rate += np.random.normal(0, 0.02, size=len(time_points))
    
    # Ensure hit rates are between 0 and 1
    lru_hit_rate = np.clip(lru_hit_rate, 0, 1)
    lfu_hit_rate = np.clip(lfu_hit_rate, 0, 1)
    arc_hit_rate = np.clip(arc_hit_rate, 0, 1)
    sa_ttl_hit_rate = np.clip(sa_ttl_hit_rate, 0, 1)
    hybrid_hit_rate = np.clip(hybrid_hit_rate, 0, 1)
    
    # Generate latency data (in milliseconds)
    # LRU
    lru_latency = 30 + 10 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    lru_latency += 15 * (time_points / time_period)
    # Add some noise
    lru_latency += np.random.normal(0, 3, size=len(time_points))
    
    # LFU
    lfu_latency = 28 + 10 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    lfu_latency += 12 * (time_points / time_period)
    # Add some noise
    lfu_latency += np.random.normal(0, 3, size=len(time_points))
    
    # ARC
    arc_latency = 25 + 8 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    arc_latency += 10 * (time_points / time_period)
    # Add some noise
    arc_latency += np.random.normal(0, 2, size=len(time_points))
    
    # Security-Aware TTL
    sa_ttl_latency = 27 + 9 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    sa_ttl_latency += 8 * (time_points / time_period)
    # Add some noise
    sa_ttl_latency += np.random.normal(0, 2, size=len(time_points))
    
    # Hybrid Cache-Graph RAG
    hybrid_latency = 22 + 7 * np.sin(2 * np.pi * time_points / 24)  # Daily cycle
    # Add some degradation over time
    hybrid_latency += 5 * (time_points / time_period)
    # Add some noise
    hybrid_latency += np.random.normal(0, 2, size=len(time_points))
    
    # Ensure latencies are positive
    lru_latency = np.maximum(lru_latency, 5)
    lfu_latency = np.maximum(lfu_latency, 5)
    arc_latency = np.maximum(arc_latency, 5)
    sa_ttl_latency = np.maximum(sa_ttl_latency, 5)
    hybrid_latency = np.maximum(hybrid_latency, 5)
    
    # Add cache refresh events
    refresh_times = np.array([6, 12, 18])  # Refresh every 6 hours
    
    return {
        'time_points': time_points,
        'refresh_times': refresh_times,
        'hit_rates': {
            'LRU': lru_hit_rate,
            'LFU': lfu_hit_rate,
            'ARC': arc_hit_rate,
            'Security-Aware TTL': sa_ttl_hit_rate,
            'Hybrid Cache-Graph RAG': hybrid_hit_rate
        },
        'latencies': {
            'LRU': lru_latency,
            'LFU': lfu_latency,
            'ARC': arc_latency,
            'Security-Aware TTL': sa_ttl_latency,
            'Hybrid Cache-Graph RAG': hybrid_latency
        }
    }

def plot_cache_performance_time_series(time_series_data, filename_prefix):
    # 1. Hit Rate Over Time
    plt.figure(figsize=(16, 10))
    
    # Convert time points to datetime for better formatting
    start_time = datetime.datetime(2025, 4, 21, 0, 0, 0)  # Start at midnight
    
    # Convert numpy values to Python native types to avoid type errors
    time_points = [float(t) for t in time_series_data['time_points']]
    refresh_times = [float(t) for t in time_series_data['refresh_times']]
    
    time_datetimes = [start_time + datetime.timedelta(hours=t) for t in time_points]
    refresh_datetimes = [start_time + datetime.timedelta(hours=t) for t in refresh_times]
    
    # Plot hit rate for each cache strategy
    strategies = list(time_series_data['hit_rates'].keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, strategy in enumerate(strategies):
        plt.plot(time_datetimes, time_series_data['hit_rates'][strategy], 
                color=colors[i], linewidth=2, label=strategy)
    
    # Mark cache refresh events
    for refresh_time in refresh_datetimes:
        plt.axvline(x=refresh_time, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time')
    plt.ylabel('Cache Hit Rate')
    plt.title('Cache Hit Rate Over Time for Different Strategies')
    
    # Format x-axis to show time
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.gcf().autofmt_xdate()
    
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}hit_rate_time_series.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Latency Over Time
    plt.figure(figsize=(16, 10))
    
    # Plot latency for each cache strategy
    for i, strategy in enumerate(strategies):
        plt.plot(time_datetimes, time_series_data['latencies'][strategy], 
                color=colors[i], linewidth=2, label=strategy)
    
    # Mark cache refresh events
    for refresh_time in refresh_datetimes:
        plt.axvline(x=refresh_time, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time')
    plt.ylabel('Latency (ms)')
    plt.title('Query Latency Over Time for Different Cache Strategies')
    
    # Format x-axis to show time
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.gcf().autofmt_xdate()
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}latency_time_series.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Recovery Visualization
    plt.figure(figsize=(16, 10))
    
    # Focus on a single refresh event
    refresh_idx = 1  # Use the second refresh event
    refresh_time = refresh_times[refresh_idx]
    
    # Convert to datetime
    refresh_datetime = start_time + datetime.timedelta(hours=refresh_time)
    
    # Define window around refresh (2 hours before, 4 hours after)
    window_start = refresh_time - 2
    window_end = refresh_time + 4
    
    # Find indices in time_points that fall within this window
    window_indices = np.where((np.array(time_points) >= window_start) & 
                             (np.array(time_points) <= window_end))[0]
    
    window_times = [time_datetimes[i] for i in window_indices]
    
    # Plot hit rate for each strategy within the window
    for i, strategy in enumerate(strategies):
        window_hit_rates = [time_series_data['hit_rates'][strategy][i] for i in window_indices]
        plt.plot(window_times, window_hit_rates, color=colors[i], linewidth=2, label=strategy)
    
    # Mark the refresh event
    plt.axvline(x=refresh_datetime, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Cache Refresh')
    
    plt.xlabel('Time')
    plt.ylabel('Cache Hit Rate')
    plt.title('Cache Performance Recovery After Refresh')
    
    # Format x-axis to show time
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(MinuteLocator(interval=30))
    plt.gcf().autofmt_xdate()
    
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}performance_recovery.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Cache Strategy Comparison
def plot_cache_strategy_comparison(time_series_data, filename_prefix):
    # Calculate average metrics for each strategy
    strategies = list(time_series_data['hit_rates'].keys())
    avg_hit_rates = {}
    avg_latencies = {}
    degradation_rates = {}
    recovery_speeds = {}
    
    # Convert numpy values to Python native types to avoid type errors
    time_points = [float(t) for t in time_series_data['time_points']]
    refresh_times = [float(t) for t in time_series_data['refresh_times']]
    
    for strategy in strategies:
        # Average hit rate
        avg_hit_rates[strategy] = float(np.mean(time_series_data['hit_rates'][strategy]))
        
        # Average latency
        avg_latencies[strategy] = float(np.mean(time_series_data['latencies'][strategy]))
        
        # Degradation rate (slope of hit rate over time)
        x = np.array(time_points)
        y = time_series_data['hit_rates'][strategy]
        degradation_rates[strategy] = float(-np.polyfit(x, y, 1)[0] * 24)  # Convert to hit rate loss per day
        
        # Recovery speed (estimated from hit rate increase after refresh)
        # Use the second refresh event
        refresh_time = refresh_times[1]
        
        # Find index closest to refresh time
        refresh_idx = np.abs(np.array(time_points) - refresh_time).argmin()
        
        # Find index 30 minutes after refresh
        recovery_idx = np.abs(np.array(time_points) - (refresh_time + 0.5)).argmin()
        
        # Calculate recovery speed as hit rate increase per hour
        if recovery_idx > refresh_idx:
            recovery_speed = (time_series_data['hit_rates'][strategy][recovery_idx] - 
                             time_series_data['hit_rates'][strategy][refresh_idx]) * 2  # Convert to per hour
            recovery_speeds[strategy] = float(max(0, recovery_speed))  # Ensure non-negative
        else:
            recovery_speeds[strategy] = 0
    
    # Create bar chart comparing all metrics
    metrics = {
        'Average Hit Rate': avg_hit_rates,
        'Average Latency (ms)': avg_latencies,
        'Degradation Rate (hit rate loss/day)': degradation_rates,
        'Recovery Speed (hit rate gain/hour)': recovery_speeds
    }
    
    # For each metric, create a bar chart
    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=(14, 10))
        
        # For latency, lower is better, so invert the values for visualization
        if 'Latency' in metric_name:
            # Normalize to 0-1 range (inverted)
            max_val = max(metric_values.values())
            min_val = min(metric_values.values())
            normalized_values = {k: 1 - (v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 
                               for k, v in metric_values.items()}
            
            # Plot original values but color bars based on normalized values
            bars = plt.bar(metric_values.keys(), metric_values.values(), width=0.6)
            
            # Color bars based on normalized values (higher is better)
            for i, (strategy, norm_val) in enumerate(normalized_values.items()):
                bars[i].set_color(plt.cm.RdYlGn(norm_val))
            
            # Add value labels on top of bars
            for i, v in enumerate(metric_values.values()):
                plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
            
            # Indicate that lower is better
            plt.figtext(0.5, 0.01, '* Lower values are better for latency', ha='center', fontsize=10, 
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
        # For degradation rate, lower is better, so invert the values for visualization
        elif 'Degradation' in metric_name:
            # Normalize to 0-1 range (inverted)
            max_val = max(metric_values.values())
            min_val = min(metric_values.values())
            normalized_values = {k: 1 - (v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 
                               for k, v in metric_values.items()}
            
            # Plot original values but color bars based on normalized values
            bars = plt.bar(metric_values.keys(), metric_values.values(), width=0.6)
            
            # Color bars based on normalized values (higher is better)
            for i, (strategy, norm_val) in enumerate(normalized_values.items()):
                bars[i].set_color(plt.cm.RdYlGn(norm_val))
            
            # Add value labels on top of bars
            for i, v in enumerate(metric_values.values()):
                plt.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
            
            # Indicate that lower is better
            plt.figtext(0.5, 0.01, '* Lower values are better for degradation rate', ha='center', fontsize=10, 
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
        else:
            # For other metrics, higher is better
            # Normalize to 0-1 range
            max_val = max(metric_values.values())
            min_val = min(metric_values.values())
            normalized_values = {k: (v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 
                               for k, v in metric_values.items()}
            
            # Plot original values but color bars based on normalized values
            bars = plt.bar(metric_values.keys(), metric_values.values(), width=0.6)
            
            # Color bars based on normalized values (higher is better)
            for i, (strategy, norm_val) in enumerate(normalized_values.items()):
                bars[i].set_color(plt.cm.RdYlGn(norm_val))
            
            # Add value labels on top of bars
            for i, v in enumerate(metric_values.values()):
                if 'Hit Rate' in metric_name:
                    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
                else:
                    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Cache Strategy')
        plt.ylabel(metric_name)
        plt.title(f'Comparison of {metric_name} Across Cache Strategies')
        
        plt.grid(True, axis='y', alpha=0.3)
        
        if 'Hit Rate' in metric_name:
            plt.ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Create a safe filename
        safe_metric_name = metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
        plt.savefig(f"{filename_prefix}comparison_{safe_metric_name}.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.close()

# Generate all the data
print("Generating cache hit rate data...")
cache_hit_rate_data = generate_cache_hit_rate_data()

print("Generating cache invalidation data...")
cache_invalidation_data = generate_cache_invalidation_data()

print("Generating cache performance time series data...")
cache_performance_data = generate_cache_performance_time_series()

# Create all the visualizations
print("Creating cache hit rate heatmaps...")
plot_cache_hit_rate_heatmaps(cache_hit_rate_data, '/home/ubuntu/research/visualizations/cache_performance/')

print("Creating cache invalidation timing diagrams...")
plot_cache_invalidation_diagrams(cache_invalidation_data, '/home/ubuntu/research/visualizations/cache_performance/')

print("Creating cache performance time series plots...")
plot_cache_performance_time_series(cache_performance_data, '/home/ubuntu/research/visualizations/cache_performance/')

print("Creating cache strategy comparison charts...")
plot_cache_strategy_comparison(cache_performance_data, '/home/ubuntu/research/visualizations/cache_performance/')

print("Cache performance visualizations generated successfully.")
