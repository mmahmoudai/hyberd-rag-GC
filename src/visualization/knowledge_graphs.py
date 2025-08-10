import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Create directory for saving visualizations
os.makedirs('/home/ubuntu/research/visualizations/knowledge_graphs', exist_ok=True)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set up matplotlib parameters for publication quality
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 14

# Define color scheme (colorblind-friendly)
colors = {
    'ip': '#1f77b4',       # blue
    'port': '#ff7f0e',     # orange
    'protocol': '#2ca02c', # green
    'service': '#d62728',  # red
    'event': '#9467bd',    # purple
    'normal': '#8c564b',   # brown
    'attack': '#e377c2',   # pink
    'highlight': '#bcbd22' # yellow-green
}

# Function to create a realistic network security knowledge graph
def create_network_security_graph():
    G = nx.DiGraph()
    
    # Add IP address nodes
    ip_addresses = [
        '192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4',
        '192.168.1.5', '10.0.0.1', '10.0.0.2', '8.8.8.8', '1.1.1.1'
    ]
    
    for ip in ip_addresses:
        is_internal = ip.startswith(('192.168', '10.0'))
        G.add_node(ip, type='ip', internal=is_internal)
    
    # Add port nodes
    ports = [22, 80, 443, 53, 3389, 445, 8080, 25]
    for port in ports:
        G.add_node(f'PORT_{port}', type='port', number=port)
    
    # Add protocol nodes
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'SMB']
    for protocol in protocols:
        G.add_node(protocol, type='protocol')
    
    # Add service nodes
    services = ['Web Server', 'DNS Server', 'SSH Service', 'Mail Server', 'File Share']
    for service in services:
        G.add_node(service, type='service')
    
    # Add event nodes
    events = [
        'Login Attempt', 'Data Transfer', 'Port Scan', 'DNS Query',
        'Connection Established', 'Connection Terminated', 'Brute Force Attempt'
    ]
    for event in events:
        is_attack = event in ['Port Scan', 'Brute Force Attempt']
        G.add_node(event, type='event', attack=is_attack)
    
    # Add edges between IP addresses (connections)
    connections = [
        ('192.168.1.1', '192.168.1.2', {'weight': 5, 'packets': 120}),
        ('192.168.1.1', '192.168.1.3', {'weight': 3, 'packets': 75}),
        ('192.168.1.2', '192.168.1.4', {'weight': 2, 'packets': 50}),
        ('192.168.1.3', '192.168.1.5', {'weight': 4, 'packets': 100}),
        ('192.168.1.5', '10.0.0.1', {'weight': 3, 'packets': 80}),
        ('10.0.0.1', '10.0.0.2', {'weight': 6, 'packets': 150}),
        ('10.0.0.2', '8.8.8.8', {'weight': 2, 'packets': 45}),
        ('192.168.1.4', '1.1.1.1', {'weight': 1, 'packets': 30})
    ]
    
    for src, dst, attrs in connections:
        G.add_edge(src, dst, **attrs, type='connects_to')
    
    # Add edges between IPs and ports
    ip_port_connections = [
        ('192.168.1.1', 'PORT_80', {'weight': 4}),
        ('192.168.1.1', 'PORT_443', {'weight': 5}),
        ('192.168.1.2', 'PORT_22', {'weight': 3}),
        ('192.168.1.3', 'PORT_3389', {'weight': 2}),
        ('192.168.1.4', 'PORT_445', {'weight': 1}),
        ('192.168.1.5', 'PORT_8080', {'weight': 3}),
        ('10.0.0.1', 'PORT_53', {'weight': 6}),
        ('10.0.0.2', 'PORT_25', {'weight': 2})
    ]
    
    for ip, port, attrs in ip_port_connections:
        G.add_edge(ip, port, **attrs, type='uses')
    
    # Add edges between ports and protocols
    port_protocol_mappings = [
        ('PORT_80', 'HTTP', {'weight': 5}),
        ('PORT_443', 'HTTPS', {'weight': 5}),
        ('PORT_22', 'SSH', {'weight': 4}),
        ('PORT_53', 'DNS', {'weight': 6}),
        ('PORT_3389', 'TCP', {'weight': 3}),
        ('PORT_445', 'SMB', {'weight': 2}),
        ('PORT_8080', 'HTTP', {'weight': 4}),
        ('PORT_25', 'TCP', {'weight': 3})
    ]
    
    for port, protocol, attrs in port_protocol_mappings:
        G.add_edge(port, protocol, **attrs, type='implements')
    
    # Add edges between protocols and services
    protocol_service_mappings = [
        ('HTTP', 'Web Server', {'weight': 5}),
        ('HTTPS', 'Web Server', {'weight': 5}),
        ('DNS', 'DNS Server', {'weight': 6}),
        ('SSH', 'SSH Service', {'weight': 4}),
        ('SMB', 'File Share', {'weight': 3}),
        ('TCP', 'Mail Server', {'weight': 2})
    ]
    
    for protocol, service, attrs in protocol_service_mappings:
        G.add_edge(protocol, service, **attrs, type='provides')
    
    # Add edges between IPs and events
    ip_event_mappings = [
        ('192.168.1.1', 'Connection Established', {'weight': 4, 'timestamp': '2025-04-21T10:15:23'}),
        ('192.168.1.2', 'Login Attempt', {'weight': 3, 'timestamp': '2025-04-21T10:16:45'}),
        ('192.168.1.3', 'Data Transfer', {'weight': 5, 'timestamp': '2025-04-21T10:18:12'}),
        ('192.168.1.4', 'Connection Terminated', {'weight': 2, 'timestamp': '2025-04-21T10:25:18'}),
        ('192.168.1.5', 'DNS Query', {'weight': 3, 'timestamp': '2025-04-21T10:30:05'}),
        ('10.0.0.1', 'Port Scan', {'weight': 6, 'timestamp': '2025-04-21T10:35:42', 'suspicious': True}),
        ('10.0.0.2', 'Brute Force Attempt', {'weight': 7, 'timestamp': '2025-04-21T10:40:19', 'suspicious': True})
    ]
    
    for ip, event, attrs in ip_event_mappings:
        G.add_edge(ip, event, **attrs, type='generates')
    
    return G

# Create the network security knowledge graph
G = create_network_security_graph()

# 1. Network Entity Relationship Graph
def visualize_network_entity_graph(G, filename):
    plt.figure(figsize=(16, 12))
    
    # Create position layout
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Draw nodes by type with different colors
    node_types = set(nx.get_node_attributes(G, 'type').values())
    
    for node_type in node_types:
        node_list = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == node_type]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=node_list,
                              node_color=colors[node_type],
                              node_size=300,
                              alpha=0.8,
                              label=f"{node_type.capitalize()} Nodes")
    
    # Draw edges with varying width based on weight
    edge_weights = [G[u][v]['weight']*0.5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, arrows=True, arrowsize=15)
    
    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add a legend
    plt.legend(loc='upper right', frameon=True, title="Node Types")
    
    plt.title("Network Security Knowledge Graph: Entity Relationships", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Edge Relationship Visualization
def visualize_edge_relationships(G, filename):
    plt.figure(figsize=(16, 12))
    
    # Create position layout
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Draw nodes with reduced size
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgray', alpha=0.6)
    
    # Draw edges by type with different colors and styles
    edge_types = set(nx.get_edge_attributes(G, 'type').values())
    
    # Define line styles for different edge types
    line_styles = {
        'connects_to': 'solid',
        'uses': 'dashed',
        'implements': 'dotted',
        'provides': 'dashdot',
        'generates': 'solid'
    }
    
    # Custom edge colors
    edge_colors = {
        'connects_to': '#1f77b4',  # blue
        'uses': '#ff7f0e',         # orange
        'implements': '#2ca02c',   # green
        'provides': '#d62728',     # red
        'generates': '#9467bd'     # purple
    }
    
    # Draw edges by type
    for edge_type in edge_types:
        edge_list = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get('type') == edge_type]
        
        # Get weights for this edge type
        widths = [G[u][v]['weight']*0.5 for u, v in edge_list]
        
        nx.draw_networkx_edges(G, pos, 
                              edgelist=edge_list,
                              width=widths,
                              alpha=0.7,
                              edge_color=edge_colors[edge_type],
                              style=line_styles[edge_type],
                              arrows=True,
                              arrowsize=15,
                              label=f"{edge_type.replace('_', ' ').capitalize()}")
    
    # Add minimal labels to nodes
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', alpha=0.7)
    
    # Add a legend for edge types
    plt.legend(loc='upper right', frameon=True, title="Edge Types")
    
    plt.title("Network Security Knowledge Graph: Edge Relationships", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Temporal Sequence Visualization
def visualize_temporal_sequence(G, filename):
    plt.figure(figsize=(16, 8))
    
    # Filter edges with timestamp attribute
    temporal_edges = [(u, v) for u, v, attrs in G.edges(data=True) if 'timestamp' in attrs]
    
    # Create a subgraph with only the nodes connected by temporal edges and their immediate neighbors
    nodes_in_temporal = set()
    for u, v in temporal_edges:
        nodes_in_temporal.add(u)
        nodes_in_temporal.add(v)
        
    # Add immediate neighbors
    neighbors = set()
    for node in nodes_in_temporal:
        neighbors.update(G.neighbors(node))
    nodes_in_temporal.update(neighbors)
    
    # Create subgraph
    subG = G.subgraph(nodes_in_temporal)
    
    # Create a custom layout that places events in chronological order
    pos = nx.spring_layout(subG, seed=42)
    
    # Adjust positions to show temporal sequence (events on a timeline)
    events = [node for node, attrs in subG.nodes(data=True) if attrs.get('type') == 'event']
    
    # Sort events by timestamp if available in the edge attributes
    event_times = {}
    for u, v, attrs in G.edges(data=True):
        if 'timestamp' in attrs and v in events:
            event_times[v] = attrs['timestamp']
    
    # Sort events by timestamp
    sorted_events = sorted(event_times.keys(), key=lambda x: event_times[x])
    
    # Adjust positions to place events in a timeline
    timeline_y = 0
    for i, event in enumerate(sorted_events):
        pos[event] = np.array([i, timeline_y])
        
        # Find connected IPs and adjust their positions
        connected_ips = []
        for u, v in G.edges():
            if v == event and G.nodes[u].get('type') == 'ip':
                connected_ips.append(u)
        
        # Position IPs above the timeline
        for j, ip in enumerate(connected_ips):
            pos[ip] = np.array([i, timeline_y + 1 + j*0.5])
    
    # Draw nodes by type
    for node_type in ['ip', 'event']:
        node_list = [node for node in subG.nodes() if subG.nodes[node].get('type') == node_type]
        
        # Highlight attack events
        if node_type == 'event':
            attack_events = [node for node in node_list if subG.nodes[node].get('attack', False)]
            normal_events = [node for node in node_list if not subG.nodes[node].get('attack', False)]
            
            nx.draw_networkx_nodes(subG, pos, 
                                  nodelist=normal_events,
                                  node_color=colors['normal'],
                                  node_size=300,
                                  alpha=0.8,
                                  label="Normal Events")
            
            nx.draw_networkx_nodes(subG, pos, 
                                  nodelist=attack_events,
                                  node_color=colors['attack'],
                                  node_size=300,
                                  alpha=0.8,
                                  label="Attack Events")
        else:
            nx.draw_networkx_nodes(subG, pos, 
                                  nodelist=node_list,
                                  node_color=colors[node_type],
                                  node_size=300,
                                  alpha=0.8,
                                  label=f"{node_type.capitalize()} Nodes")
    
    # Draw temporal edges
    temporal_edge_list = [(u, v) for u, v in subG.edges() if (u, v) in temporal_edges]
    nx.draw_networkx_edges(subG, pos, 
                          edgelist=temporal_edge_list,
                          width=2,
                          alpha=0.7,
                          edge_color='black',
                          style='solid',
                          arrows=True,
                          arrowsize=15,
                          label="Temporal Sequence")
    
    # Add labels to nodes
    nx.draw_networkx_labels(subG, pos, font_size=10, font_family='sans-serif')
    
    # Add timestamps as edge labels
    edge_labels = {(u, v): attrs['timestamp'].split('T')[1] 
                  for u, v, attrs in G.edges(data=True) 
                  if 'timestamp' in attrs and (u, v) in temporal_edge_list}
    
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=8)
    
    # Add a legend
    plt.legend(loc='upper right', frameon=True)
    
    plt.title("Network Security Knowledge Graph: Temporal Event Sequence", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Subgraph Extraction Visualization
def visualize_subgraph_extraction(G, filename, query="Port Scan"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create position layout for the full graph
    pos_full = nx.spring_layout(G, seed=42, k=0.3)
    
    # Draw the full graph in the first subplot
    ax1.set_title("Original Knowledge Graph", fontsize=16)
    
    # Draw nodes with reduced opacity
    for node_type in set(nx.get_node_attributes(G, 'type').values()):
        node_list = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == node_type]
        nx.draw_networkx_nodes(G, pos_full, 
                              nodelist=node_list,
                              node_color=colors[node_type],
                              node_size=200,
                              alpha=0.5,
                              ax=ax1)
    
    # Draw edges with reduced opacity
    nx.draw_networkx_edges(G, pos_full, alpha=0.3, arrows=True, arrowsize=10, ax=ax1)
    
    # Add minimal labels
    nx.draw_networkx_labels(G, pos_full, font_size=8, font_family='sans-serif', ax=ax1)
    
    # Now create and visualize the extracted subgraph based on the query
    # For this example, we'll extract a subgraph related to the query (e.g., "Port Scan")
    
    # Find the query node
    query_node = None
    for node, attrs in G.nodes(data=True):
        if node == query:
            query_node = node
            break
    
    if not query_node:
        # If exact query not found, find a node that contains the query string
        for node, attrs in G.nodes(data=True):
            if isinstance(node, str) and query.lower() in node.lower():
                query_node = node
                break
    
    # If still not found, use a default node
    if not query_node:
        query_node = list(G.nodes())[0]
    
    # Extract subgraph (2-hop neighborhood)
    neighbors_1hop = list(G.neighbors(query_node))
    neighbors_2hop = []
    for neighbor in neighbors_1hop:
        neighbors_2hop.extend(list(G.neighbors(neighbor)))
    
    subgraph_nodes = [query_node] + neighbors_1hop + neighbors_2hop
    subgraph_nodes = list(set(subgraph_nodes))  # Remove duplicates
    
    subG = G.subgraph(subgraph_nodes)
    
    # Create position layout for the subgraph
    pos_sub = nx.spring_layout(subG, seed=42, k=0.4)
    
    # Draw the subgraph in the second subplot
    ax2.set_title(f"Extracted Subgraph for Query: '{query}'", fontsize=16)
    
    # Draw nodes by type with higher opacity
    for node_type in set(nx.get_node_attributes(subG, 'type').values()):
        node_list = [node for node, attrs in subG.nodes(data=True) if attrs.get('type') == node_type]
        
        # Highlight the query node
        if query_node in node_list:
            # Remove from regular list
            node_list.remove(query_node)
            
            # Draw separately with highlight
            nx.draw_networkx_nodes(subG, pos_sub, 
                                  nodelist=[query_node],
                                  node_color=colors['highlight'],
                                  node_size=500,
                                  alpha=1.0,
                                  ax=ax2,
                                  label="Query Node")
        
        nx.draw_networkx_nodes(subG, pos_sub, 
                              nodelist=node_list,
                              node_color=colors[node_type],
                              node_size=300,
                              alpha=0.8,
                              ax=ax2,
                              label=f"{node_type.capitalize()} Nodes" if node_list else "")
    
    # Draw edges with full opacity
    nx.draw_networkx_edges(subG, pos_sub, alpha=0.7, arrows=True, arrowsize=15, ax=ax2)
    
    # Add labels
    nx.draw_networkx_labels(subG, pos_sub, font_size=10, font_family='sans-serif', ax=ax2)
    
    # Add a legend to the second subplot
    ax2.legend(loc='upper right', frameon=True)
    
    # Turn off axis for both subplots
    ax1.axis('off')
    ax2.axis('off')
    
    plt.suptitle("Graph RAG: Subgraph Extraction for Network Security Analysis", fontsize=20)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Relevance Heat Map Visualization
def visualize_relevance_heatmap(G, filename, query="Brute Force Attempt"):
    plt.figure(figsize=(16, 12))
    
    # Find the query node
    query_node = None
    for node, attrs in G.nodes(data=True):
        if node == query:
            query_node = node
            break
    
    if not query_node:
        # If exact query not found, find a node that contains the query string
        for node, attrs in G.nodes(data=True):
            if isinstance(node, str) and query.lower() in node.lower():
                query_node = node
                break
    
    # If still not found, use a default node
    if not query_node:
        query_node = list(G.nodes())[0]
    
    # Extract subgraph (2-hop neighborhood)
    neighbors_1hop = list(G.neighbors(query_node))
    neighbors_2hop = []
    for neighbor in neighbors_1hop:
        neighbors_2hop.extend(list(G.neighbors(neighbor)))
    
    subgraph_nodes = [query_node] + neighbors_1hop + neighbors_2hop
    subgraph_nodes = list(set(subgraph_nodes))  # Remove duplicates
    
    subG = G.subgraph(subgraph_nodes)
    
    # Create position layout
    pos = nx.spring_layout(subG, seed=42, k=0.4)
    
    # Calculate relevance scores (simulated)
    # In a real system, this would be based on semantic similarity, graph distance, etc.
    relevance_scores = {}
    
    # Query node has maximum relevance
    relevance_scores[query_node] = 1.0
    
    # First-hop neighbors have high relevance
    for node in neighbors_1hop:
        # Simulate relevance based on edge weight if available
        if 'weight' in G[query_node][node]:
            relevance_scores[node] = 0.7 + (G[query_node][node]['weight'] / 10)
        else:
            relevance_scores[node] = 0.7
    
    # Second-hop neighbors have lower relevance
    for node in neighbors_2hop:
        if node not in neighbors_1hop and node != query_node:
            # Find the connecting first-hop neighbor with highest weight
            max_weight = 0
            for n1 in neighbors_1hop:
                if subG.has_edge(n1, node) and 'weight' in subG[n1][node]:
                    max_weight = max(max_weight, subG[n1][node]['weight'])
            
            relevance_scores[node] = 0.3 + (max_weight / 20)
    
    # Normalize relevance scores between 0 and 1
    min_score = min(relevance_scores.values())
    max_score = max(relevance_scores.values())
    
    if max_score > min_score:  # Avoid division by zero
        for node in relevance_scores:
            relevance_scores[node] = (relevance_scores[node] - min_score) / (max_score - min_score)
    
    # Draw nodes with color intensity based on relevance
    node_colors = []
    node_sizes = []
    
    for node in subG.nodes():
        relevance = relevance_scores.get(node, 0.1)
        
        # Adjust color intensity based on relevance
        if node == query_node:
            node_colors.append(colors['highlight'])
            node_sizes.append(500)
        else:
            # Get node type color
            base_color = colors[subG.nodes[node].get('type', 'ip')]
            node_colors.append(base_color)
            node_sizes.append(300 * relevance + 100)  # Size also indicates relevance
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(subG, pos, 
                                  node_size=node_sizes,
                                  node_color=node_colors,
                                  alpha=0.8)
    
    # Draw edges with width based on relevance of connected nodes
    edge_widths = []
    for u, v in subG.edges():
        source_relevance = relevance_scores.get(u, 0.1)
        target_relevance = relevance_scores.get(v, 0.1)
        edge_widths.append(1 + 3 * (source_relevance + target_relevance) / 2)
    
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.6, arrows=True, arrowsize=15)
    
    # Add labels to nodes
    nx.draw_networkx_labels(subG, pos, font_size=10, font_family='sans-serif')
    
    # Add a colorbar to show relevance scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Relevance Score', fontsize=14)
    
    plt.title(f"Graph RAG: Relevance Heat Map for Query '{query}'", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
visualize_network_entity_graph(G, '/home/ubuntu/research/visualizations/knowledge_graphs/network_entity_graph.svg')
visualize_edge_relationships(G, '/home/ubuntu/research/visualizations/knowledge_graphs/edge_relationships.svg')
visualize_temporal_sequence(G, '/home/ubuntu/research/visualizations/knowledge_graphs/temporal_sequence.svg')
visualize_subgraph_extraction(G, '/home/ubuntu/research/visualizations/knowledge_graphs/subgraph_extraction.svg', query="Port Scan")
visualize_relevance_heatmap(G, '/home/ubuntu/research/visualizations/knowledge_graphs/relevance_heatmap.svg', query="Brute Force Attempt")

print("Knowledge graph visualizations generated successfully.")
