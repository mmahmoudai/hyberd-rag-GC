"""
Utility functions for network packet processing and data transformation.
"""

import re
import json
import hashlib
import ipaddress
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime

def format_packet_data(packet_data: Dict[str, Any]) -> str:
    """
    Format packet data into a structured text representation for embedding.
    
    Args:
        packet_data: Dictionary containing packet information
        
    Returns:
        Formatted text representation
    """
    template = """
    Flow ID: {flow_id}
    Source IP: {src_ip}
    Destination IP: {dst_ip}
    Source Port: {src_port}
    Destination Port: {dst_port}
    Protocol: {protocol}
    Timestamp: {timestamp}
    Duration: {duration}
    Bytes Sent: {bytes_sent}
    Bytes Received: {bytes_received}
    Packets Sent: {packets_sent}
    Packets Received: {packets_received}
    Additional Info: {additional_info}
    """
    
    # Extract values with defaults for missing keys
    formatted = template.format(
        flow_id=packet_data.get('flow_id', 'unknown'),
        src_ip=packet_data.get('src_ip', 'unknown'),
        dst_ip=packet_data.get('dst_ip', 'unknown'),
        src_port=packet_data.get('src_port', 'unknown'),
        dst_port=packet_data.get('dst_port', 'unknown'),
        protocol=packet_data.get('protocol', 'unknown'),
        timestamp=packet_data.get('timestamp', 'unknown'),
        duration=packet_data.get('duration', 'unknown'),
        bytes_sent=packet_data.get('bytes_sent', 'unknown'),
        bytes_received=packet_data.get('bytes_received', 'unknown'),
        packets_sent=packet_data.get('packets_sent', 'unknown'),
        packets_received=packet_data.get('packets_received', 'unknown'),
        additional_info=json.dumps(
            {k: v for k, v in packet_data.items() 
             if k not in ['flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 
                         'protocol', 'timestamp', 'duration', 'bytes_sent', 
                         'bytes_received', 'packets_sent', 'packets_received']}
        )
    )
    
    return formatted.strip()

def generate_flow_id(packet_data: Dict[str, Any]) -> str:
    """
    Generate a unique flow ID based on packet data.
    
    Args:
        packet_data: Dictionary containing packet information
        
    Returns:
        Unique flow ID
    """
    # Create a unique identifier based on 5-tuple
    components = [
        str(packet_data.get('src_ip', '')),
        str(packet_data.get('dst_ip', '')),
        str(packet_data.get('src_port', '')),
        str(packet_data.get('dst_port', '')),
        str(packet_data.get('protocol', ''))
    ]
    
    # Add timestamp if available for further uniqueness
    if 'timestamp' in packet_data:
        components.append(str(packet_data['timestamp']))
    
    # Create hash
    flow_string = "_".join(components)
    return hashlib.md5(flow_string.encode()).hexdigest()

def is_private_ip(ip_str: str) -> bool:
    """
    Check if an IP address is private.
    
    Args:
        ip_str: IP address string
        
    Returns:
        True if private, False otherwise
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private
    except ValueError:
        return False

def extract_protocol_features(packet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract protocol-specific features from packet data.
    
    Args:
        packet_data: Dictionary containing packet information
        
    Returns:
        Dictionary of protocol-specific features
    """
    features = {}
    protocol = packet_data.get('protocol', '').lower()
    
    # HTTP features
    if protocol == 'http' or packet_data.get('dst_port') == 80:
        if 'http_method' in packet_data:
            features['http_method'] = packet_data['http_method']
        if 'http_uri' in packet_data:
            features['http_uri'] = packet_data['http_uri']
        if 'http_version' in packet_data:
            features['http_version'] = packet_data['http_version']
        if 'http_user_agent' in packet_data:
            features['http_user_agent'] = packet_data['http_user_agent']
    
    # DNS features
    elif protocol == 'dns' or packet_data.get('dst_port') == 53:
        if 'dns_query' in packet_data:
            features['dns_query'] = packet_data['dns_query']
        if 'dns_answer' in packet_data:
            features['dns_answer'] = packet_data['dns_answer']
        if 'dns_query_type' in packet_data:
            features['dns_query_type'] = packet_data['dns_query_type']
    
    # SSH features
    elif protocol == 'ssh' or packet_data.get('dst_port') == 22:
        if 'ssh_version' in packet_data:
            features['ssh_version'] = packet_data['ssh_version']
    
    # TLS/SSL features
    elif protocol in ['tls', 'ssl'] or packet_data.get('dst_port') == 443:
        if 'tls_version' in packet_data:
            features['tls_version'] = packet_data['tls_version']
        if 'tls_cipher_suite' in packet_data:
            features['tls_cipher_suite'] = packet_data['tls_cipher_suite']
        if 'tls_sni' in packet_data:
            features['tls_sni'] = packet_data['tls_sni']
    
    return features

def calculate_statistical_features(flow_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate statistical features from a list of packets in a flow.
    
    Args:
        flow_data: List of packet dictionaries in a flow
        
    Returns:
        Dictionary of statistical features
    """
    if not flow_data:
        return {}
    
    # Extract packet sizes and inter-arrival times
    packet_sizes = []
    inter_arrival_times = []
    prev_time = None
    
    for packet in flow_data:
        # Packet size
        if 'length' in packet:
            packet_sizes.append(packet['length'])
        
        # Inter-arrival time
        if 'timestamp' in packet:
            current_time = packet['timestamp']
            if prev_time is not None:
                # Calculate time difference
                if isinstance(current_time, str):
                    try:
                        current_dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
                        prev_dt = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                        diff = (current_dt - prev_dt).total_seconds()
                    except ValueError:
                        diff = 0
                else:
                    diff = current_time - prev_time
                
                inter_arrival_times.append(diff)
            
            prev_time = current_time
    
    # Calculate statistics
    features = {}
    
    # Packet size statistics
    if packet_sizes:
        features['min_packet_size'] = min(packet_sizes)
        features['max_packet_size'] = max(packet_sizes)
        features['mean_packet_size'] = np.mean(packet_sizes)
        features['std_packet_size'] = np.std(packet_sizes)
    
    # Inter-arrival time statistics
    if inter_arrival_times:
        features['min_iat'] = min(inter_arrival_times)
        features['max_iat'] = max(inter_arrival_times)
        features['mean_iat'] = np.mean(inter_arrival_times)
        features['std_iat'] = np.std(inter_arrival_times)
    
    # Flow duration
    if len(flow_data) >= 2 and 'timestamp' in flow_data[0] and 'timestamp' in flow_data[-1]:
        first_time = flow_data[0]['timestamp']
        last_time = flow_data[-1]['timestamp']
        
        if isinstance(first_time, str):
            try:
                first_dt = datetime.fromisoformat(first_time.replace('Z', '+00:00'))
                last_dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                duration = (last_dt - first_dt).total_seconds()
            except ValueError:
                duration = 0
        else:
            duration = last_time - first_time
        
        features['flow_duration'] = duration
    
    return features

def extract_threat_indicators(packet_data: Dict[str, Any]) -> List[str]:
    """
    Extract potential threat indicators from packet data.
    
    Args:
        packet_data: Dictionary containing packet information
        
    Returns:
        List of potential threat indicators
    """
    indicators = []
    
    # Check for unusual ports
    suspicious_ports = [4444, 31337, 1337, 9001, 9002, 6667]  # Example suspicious ports
    if packet_data.get('dst_port') in suspicious_ports:
        indicators.append(f"Suspicious destination port: {packet_data.get('dst_port')}")
    
    # Check for unusual protocols or combinations
    if packet_data.get('protocol') == 'icmp' and packet_data.get('length', 0) > 1000:
        indicators.append("Large ICMP packet (possible tunneling)")
    
    # Check for unusual HTTP user agents
    user_agent = packet_data.get('http_user_agent', '')
    if user_agent and ('curl' in user_agent.lower() or 'wget' in user_agent.lower()):
        indicators.append(f"Command-line web client: {user_agent}")
    
    # Check for base64 encoded data in unusual places
    base64_pattern = r'[A-Za-z0-9+/]{30,}={0,2}'
    for key, value in packet_data.items():
        if isinstance(value, str) and re.search(base64_pattern, value):
            indicators.append(f"Possible base64 encoded data in {key}")
    
    # Check for potential command and control domains in DNS
    if 'dns_query' in packet_data:
        domain = packet_data['dns_query']
        if domain:
            # Check for algorithmically generated domain names
            if len(domain) > 30 or domain.count('.') > 3:
                indicators.append(f"Potential DGA domain: {domain}")
            
            # Check for unusual TLDs
            unusual_tlds = ['.xyz', '.top', '.club', '.pw']
            if any(domain.endswith(tld) for tld in unusual_tlds):
                indicators.append(f"Unusual TLD in domain: {domain}")
    
    return indicators
