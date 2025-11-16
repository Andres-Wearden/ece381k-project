"""Network topology builders for multi-agent systems"""

import networkx as nx
import numpy as np
from typing import Dict, Any, Optional


def available_topologies():
    """Return list of available topologies"""
    return ['star', 'cascade', 'feedback_rewired', 'ring', 'mesh', 'small_world', 'scale_free', 'tree', 'grid']


def build_star_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build star topology with central hub
    
    Args:
        n_agents: Number of agents
        hub_id: ID of hub node (default: 0)
        bidirectional: Whether edges are bidirectional (default: True)
        
    Returns:
        Directed weighted graph
    """
    G = nx.DiGraph()
    hub_id = kwargs.get('hub_id', 0)
    bidirectional = kwargs.get('bidirectional', True)
    
    # Add nodes
    G.add_nodes_from(range(n_agents))
    
    # Connect all nodes to hub
    for i in range(n_agents):
        if i != hub_id:
            # Spoke to hub
            weight = np.random.uniform(0.5, 1.0)
            G.add_edge(i, hub_id, weight=weight, delay=0)
            
            if bidirectional:
                # Hub to spoke
                weight = np.random.uniform(0.5, 1.0)
                G.add_edge(hub_id, i, weight=weight, delay=0)
    
    return G


def build_cascade_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build cascade/chain topology
    
    Args:
        n_agents: Number of agents
        skip_connections: Add skip connections (default: False)
        
    Returns:
        Directed weighted graph
    """
    G = nx.DiGraph()
    skip_connections = kwargs.get('skip_connections', False)
    
    # Add nodes
    G.add_nodes_from(range(n_agents))
    
    # Create chain
    for i in range(n_agents - 1):
        weight = np.random.uniform(0.5, 1.0)
        delay = np.random.randint(0, 3) if kwargs.get('random_delays', False) else 0
        G.add_edge(i, i + 1, weight=weight, delay=delay)
    
    # Optional: add backward edges
    if kwargs.get('bidirectional', False):
        for i in range(1, n_agents):
            weight = np.random.uniform(0.3, 0.7)
            delay = np.random.randint(0, 3) if kwargs.get('random_delays', False) else 0
            G.add_edge(i, i - 1, weight=weight, delay=delay)
    
    # Optional: skip connections
    if skip_connections:
        for i in range(n_agents - 2):
            if np.random.rand() > 0.5:
                weight = np.random.uniform(0.3, 0.6)
                delay = np.random.randint(0, 4) if kwargs.get('random_delays', False) else 0
                G.add_edge(i, i + 2, weight=weight, delay=delay)
    
    return G


def build_feedback_rewired_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build feedback-rewired topology
    Starts with a base topology and adds feedback loops with rewiring
    
    Args:
        n_agents: Number of agents
        rewire_prob: Probability of rewiring (default: 0.3)
        feedback_prob: Probability of adding feedback edge (default: 0.4)
        base_topology: Base topology ('ring', 'mesh', 'random') (default: 'ring')
        
    Returns:
        Directed weighted graph
    """
    rewire_prob = kwargs.get('rewire_prob', 0.3)
    feedback_prob = kwargs.get('feedback_prob', 0.4)
    base_topology = kwargs.get('base_topology', 'ring')
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n_agents))
    
    # Create base topology
    if base_topology == 'ring':
        # Directed ring
        for i in range(n_agents):
            weight = np.random.uniform(0.5, 1.0)
            delay = 0
            G.add_edge(i, (i + 1) % n_agents, weight=weight, delay=delay)
            
    elif base_topology == 'mesh':
        # Partial mesh
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if np.random.rand() > 0.6:  # Sparse connections
                    weight = np.random.uniform(0.5, 1.0)
                    delay = 0
                    G.add_edge(i, j, weight=weight, delay=delay)
                    
    elif base_topology == 'random':
        # Random directed graph
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and np.random.rand() < 0.2:
                    weight = np.random.uniform(0.5, 1.0)
                    delay = 0
                    G.add_edge(i, j, weight=weight, delay=delay)
    
    # Rewiring phase
    edges = list(G.edges())
    for u, v in edges:
        if np.random.rand() < rewire_prob:
            # Remove edge and rewire
            edge_data = G[u][v]
            G.remove_edge(u, v)
            
            # Find new target
            possible_targets = [n for n in G.nodes() if n != u and not G.has_edge(u, n)]
            if possible_targets:
                new_v = np.random.choice(possible_targets)
                G.add_edge(u, new_v, **edge_data)
    
    # Add feedback edges
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            
            # Add feedback if there's a forward path
            if G.has_edge(u, v) and not G.has_edge(v, u):
                if np.random.rand() < feedback_prob:
                    weight = np.random.uniform(0.3, 0.7)
                    delay = np.random.randint(1, 4) if kwargs.get('random_delays', True) else 1
                    G.add_edge(v, u, weight=weight, delay=delay)
    
    # Ensure connectivity
    if not nx.is_weakly_connected(G):
        components = list(nx.weakly_connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            weight = np.random.uniform(0.5, 1.0)
            G.add_edge(u, v, weight=weight, delay=0)
    
    return G


def build_ring_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build ring topology - circular connections
    
    Args:
        n_agents: Number of agents
        bidirectional: Whether edges are bidirectional (default: True)
        
    Returns:
        Directed weighted graph
    """
    G = nx.DiGraph()
    bidirectional = kwargs.get('bidirectional', True)
    
    # Add nodes
    G.add_nodes_from(range(n_agents))
    
    # Create ring connections
    for i in range(n_agents):
        next_node = (i + 1) % n_agents
        weight = np.random.uniform(0.5, 1.0)
        delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
        G.add_edge(i, next_node, weight=weight, delay=delay)
        
        if bidirectional:
            # Add reverse edge
            weight = np.random.uniform(0.5, 1.0)
            delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
            G.add_edge(next_node, i, weight=weight, delay=delay)
    
    return G


def build_mesh_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build mesh/fully connected topology - all nodes connected to all others
    
    Args:
        n_agents: Number of agents
        bidirectional: Whether edges are bidirectional (default: True)
        edge_probability: Probability of including each edge (default: 1.0 for full mesh)
        
    Returns:
        Directed weighted graph
    """
    G = nx.DiGraph()
    bidirectional = kwargs.get('bidirectional', True)
    edge_probability = kwargs.get('edge_probability', 1.0)
    
    # Add nodes
    G.add_nodes_from(range(n_agents))
    
    # Connect all nodes to all others
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                if np.random.rand() <= edge_probability:
                    weight = np.random.uniform(0.5, 1.0)
                    delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
                    G.add_edge(i, j, weight=weight, delay=delay)
                    
                    if not bidirectional:
                        # For directed mesh, only add one direction
                        pass
                    # If bidirectional, we already added the edge above
    
    return G


def build_small_world_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build small-world topology using Watts-Strogatz model
    
    Args:
        n_agents: Number of agents
        k: Each node is connected to k nearest neighbors (default: 4)
        rewire_prob: Probability of rewiring each edge (default: 0.3)
        
    Returns:
        Directed weighted graph
    """
    k = kwargs.get('k', 4)
    rewire_prob = kwargs.get('rewire_prob', 0.3)
    
    # Use NetworkX Watts-Strogatz generator (creates undirected graph)
    # We'll convert to directed
    G_undirected = nx.watts_strogatz_graph(n_agents, k, rewire_prob, seed=42)
    
    # Convert to directed graph with weights and delays
    G = nx.DiGraph()
    G.add_nodes_from(range(n_agents))
    
    for u, v in G_undirected.edges():
        # Add edge in both directions for small-world
        weight = np.random.uniform(0.5, 1.0)
        delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
        G.add_edge(u, v, weight=weight, delay=delay)
        
        # Add reverse edge
        weight = np.random.uniform(0.5, 1.0)
        delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
        G.add_edge(v, u, weight=weight, delay=delay)
    
    return G


def build_scale_free_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build scale-free topology using Barabási–Albert model
    
    Args:
        n_agents: Number of agents
        m: Number of edges to attach from a new node to existing nodes (default: 2)
        
    Returns:
        Directed weighted graph
    """
    m = kwargs.get('m', 2)
    
    # Use NetworkX Barabási–Albert generator (creates undirected graph)
    # We'll convert to directed
    G_undirected = nx.barabasi_albert_graph(n_agents, m, seed=42)
    
    # Convert to directed graph with weights and delays
    G = nx.DiGraph()
    G.add_nodes_from(range(n_agents))
    
    for u, v in G_undirected.edges():
        # For scale-free, we'll make it directed but add both directions
        # to ensure connectivity
        weight = np.random.uniform(0.5, 1.0)
        delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
        G.add_edge(u, v, weight=weight, delay=delay)
        
        # Add reverse edge
        weight = np.random.uniform(0.5, 1.0)
        delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
        G.add_edge(v, u, weight=weight, delay=delay)
    
    return G


def build_tree_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build tree topology - hierarchical structure
    
    Args:
        n_agents: Number of agents
        branching_factor: Number of children per node (default: 2 for binary tree)
        
    Returns:
        Directed weighted graph
    """
    G = nx.DiGraph()
    branching_factor = kwargs.get('branching_factor', 2)
    
    # Add nodes
    G.add_nodes_from(range(n_agents))
    
    # Build tree structure
    # Root is node 0
    # For node i, children are at indices: i*branching_factor + 1, ..., i*branching_factor + branching_factor
    for i in range(n_agents):
        for j in range(1, branching_factor + 1):
            child_idx = i * branching_factor + j
            if child_idx < n_agents:
                weight = np.random.uniform(0.5, 1.0)
                delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
                G.add_edge(i, child_idx, weight=weight, delay=delay)
                
                # Add reverse edge for bidirectional communication
                if kwargs.get('bidirectional', True):
                    weight = np.random.uniform(0.3, 0.7)
                    delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
                    G.add_edge(child_idx, i, weight=weight, delay=delay)
    
    return G


def build_grid_topology(n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build grid/lattice topology - 2D regular grid
    
    Args:
        n_agents: Number of agents
        rows: Number of rows (auto-calculated if not provided)
        cols: Number of columns (auto-calculated if not provided)
        
    Returns:
        Directed weighted graph
    """
    G = nx.DiGraph()
    
    # Calculate grid dimensions
    if 'rows' in kwargs and 'cols' in kwargs:
        rows = kwargs['rows']
        cols = kwargs['cols']
        if rows * cols != n_agents:
            raise ValueError(f"rows * cols ({rows * cols}) must equal n_agents ({n_agents})")
    else:
        # Auto-calculate dimensions (prefer square-ish grid)
        rows = int(np.sqrt(n_agents))
        cols = (n_agents + rows - 1) // rows  # Ceiling division
    
    # Add nodes
    G.add_nodes_from(range(n_agents))
    
    # Create grid connections
    for i in range(rows):
        for j in range(cols):
            node_idx = i * cols + j
            if node_idx >= n_agents:
                break
            
            # Connect to right neighbor
            if j < cols - 1 and (i * cols + j + 1) < n_agents:
                right_idx = i * cols + j + 1
                weight = np.random.uniform(0.5, 1.0)
                delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
                G.add_edge(node_idx, right_idx, weight=weight, delay=delay)
                
                if kwargs.get('bidirectional', True):
                    weight = np.random.uniform(0.5, 1.0)
                    delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
                    G.add_edge(right_idx, node_idx, weight=weight, delay=delay)
            
            # Connect to bottom neighbor
            if i < rows - 1 and ((i + 1) * cols + j) < n_agents:
                bottom_idx = (i + 1) * cols + j
                weight = np.random.uniform(0.5, 1.0)
                delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
                G.add_edge(node_idx, bottom_idx, weight=weight, delay=delay)
                
                if kwargs.get('bidirectional', True):
                    weight = np.random.uniform(0.5, 1.0)
                    delay = np.random.randint(0, 2) if kwargs.get('random_delays', False) else 0
                    G.add_edge(bottom_idx, node_idx, weight=weight, delay=delay)
    
    return G


def build_topology(topology_name: str, n_agents: int, **kwargs) -> nx.DiGraph:
    """
    Build network topology
    
    Args:
        topology_name: Name of topology ('star', 'cascade', 'feedback_rewired', 'ring', 'mesh', 'small_world', 'scale_free', 'tree', 'grid')
        n_agents: Number of agents
        **kwargs: Additional topology-specific parameters
        
    Returns:
        Directed weighted graph
    """
    builders = {
        'star': build_star_topology,
        'cascade': build_cascade_topology,
        'feedback_rewired': build_feedback_rewired_topology,
        'ring': build_ring_topology,
        'mesh': build_mesh_topology,
        'small_world': build_small_world_topology,
        'scale_free': build_scale_free_topology,
        'tree': build_tree_topology,
        'grid': build_grid_topology
    }
    
    if topology_name not in builders:
        raise ValueError(f"Unknown topology: {topology_name}. Available: {list(builders.keys())}")
    
    return builders[topology_name](n_agents, **kwargs)

