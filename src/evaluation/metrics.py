"""Evaluation metrics for multi-agent benchmarking"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate prediction accuracy
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(labels, predictions)


def calculate_robustness(accuracy_history: List[float], 
                         failure_history: List[int],
                         network: Optional[nx.DiGraph] = None) -> float:
    """
    Calculate system robustness using a comprehensive metric that considers:
    1. Performance degradation during failures (resilience)
    2. Network connectivity under failures (topology resilience)
    3. Failure severity and frequency
    
    Based on network robustness literature:
    - Robustness measures how well a system maintains functionality under perturbations
    - Should account for both performance (accuracy) and structure (connectivity)
    
    Args:
        accuracy_history: List of accuracy values over time
        failure_history: List of failed node counts over time
        network: Optional network graph for topology-aware robustness
        
    Returns:
        Robustness score (0-1, higher is better)
    """
    if len(accuracy_history) < 2:
        return 1.0
    
    accuracy_mean = np.mean(accuracy_history)
    if accuracy_mean <= 0:
        return 0.0
    
    n_agents = max(failure_history) if failure_history else len(accuracy_history)
    if n_agents == 0:
        n_agents = 1
    
    # ========== COMPONENT 1: Performance Resilience ==========
    # How well does accuracy maintain during failures?
    performance_resilience = 1.0
    
    if max(failure_history) > 0:
        failure_indices = [i for i, f in enumerate(failure_history) if f > 0]
        normal_indices = [i for i, f in enumerate(failure_history) if f == 0]
        
        if failure_indices and normal_indices:
            acc_with_failures = [accuracy_history[i] for i in failure_indices]
            acc_without_failures = [accuracy_history[i] for i in normal_indices]
            
            avg_with_failures = np.mean(acc_with_failures)
            avg_without_failures = np.mean(acc_without_failures)
            
            if avg_without_failures > 0:
                # Performance ratio: how well accuracy is maintained
                performance_resilience = avg_with_failures / avg_without_failures
            else:
                performance_resilience = 0.0
        elif failure_indices:
            # Only failures - compare to mean
            avg_with_failures = np.mean([accuracy_history[i] for i in failure_indices])
            performance_resilience = avg_with_failures / accuracy_mean if accuracy_mean > 0 else 0.0
    
    # ========== COMPONENT 2: Failure Impact ==========
    # Penalize based on failure severity and frequency
    failure_impact = 0.0
    
    if max(failure_history) > 0:
        # Average fraction of nodes that failed
        avg_failure_fraction = np.mean([f / n_agents for f in failure_history if f > 0])
        # Frequency of failures
        failure_frequency = sum(1 for f in failure_history if f > 0) / len(failure_history)
        # Maximum concurrent failures
        max_failures = max(failure_history)
        max_failure_fraction = max_failures / n_agents if n_agents > 0 else 0
        
        # Combined failure impact (weighted)
        # More failures = higher impact, but topology can mitigate this
        failure_impact = 0.4 * avg_failure_fraction + 0.3 * failure_frequency + 0.3 * max_failure_fraction
        failure_impact = min(1.0, failure_impact)
    
    # ========== COMPONENT 3: Network Topology Resilience ==========
    # If network is provided, calculate topology-based resilience
    topology_resilience = 1.0
    
    if network is not None and max(failure_history) > 0:
        try:
            # Calculate network metrics that indicate robustness
            # 1. Average shortest path length (lower is better for resilience)
            if nx.is_strongly_connected(network) or nx.is_connected(network.to_undirected()):
                try:
                    avg_path_length = nx.average_shortest_path_length(network.to_undirected() if network.is_directed() else network)
                    # Normalize: shorter paths = more resilient (inverse relationship)
                    # Typical range: 1 (mesh) to ~N/2 (chain), normalize to 0-1
                    max_path = len(network.nodes()) - 1
                    topology_resilience = 1.0 - (avg_path_length - 1) / max_path if max_path > 1 else 1.0
                    topology_resilience = max(0.0, min(1.0, topology_resilience))
                except:
                    pass
            
            # 2. Network density (higher density = more redundant paths = more robust)
            density = nx.density(network)
            # Density ranges from 0 to 1, use it directly as a resilience factor
            topology_resilience = 0.6 * topology_resilience + 0.4 * density
        except:
            # If network analysis fails, use default
            topology_resilience = 0.8  # Moderate resilience assumption
    
    # ========== COMPONENT 4: Stability ==========
    # How stable is accuracy over time?
    accuracy_cv = np.std(accuracy_history) / accuracy_mean if accuracy_mean > 0 else 1.0
    stability = 1.0 / (1.0 + 3.0 * accuracy_cv)  # Moderate sensitivity to variation
    
    # ========== COMBINED ROBUSTNESS ==========
    # Robustness = Performance Resilience × (1 - Failure Impact) × Topology Resilience × Stability
    # This formula:
    # - Rewards systems that maintain accuracy during failures (performance_resilience)
    # - Penalizes systems with high failure impact
    # - Accounts for network topology structure (topology_resilience)
    # - Considers stability over time
    
    # If no failures, robustness depends on stability and topology
    if max(failure_history) == 0:
        robustness = stability * topology_resilience * 0.95  # Slight penalty for no failure testing
    else:
        # With failures: combine all factors
        # Performance resilience is primary, but failure impact reduces it
        # Topology resilience and stability modulate the result
        robustness = performance_resilience * (1.0 - 0.3 * failure_impact) * topology_resilience * stability
    
    # Ensure reasonable range
    return max(0.0, min(1.0, robustness))


def calculate_error_depth(agents: List, network: nx.DiGraph, 
                          test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate error depth: how errors propagate through network
    
    Measures average path length from incorrect predictions to their sources
    
    Args:
        agents: List of agents
        network: Communication network
        test_data: Test data
        test_labels: True labels
        
    Returns:
        Dictionary with error depth metrics
    """
    # Get predictions from each agent
    agent_predictions = {}
    agent_errors = {}
    
    for agent in agents:
        if not agent.failed:
            preds = agent.predict(test_data)
            agent_predictions[agent.agent_id] = preds
            errors = preds != test_labels
            agent_errors[agent.agent_id] = errors
    
    if not agent_predictions:
        return {'mean_depth': 0.0, 'max_depth': 0.0, 'error_rate': 1.0}
    
    # Calculate shortest paths in network
    try:
        all_paths_lengths = dict(nx.all_pairs_shortest_path_length(network))
    except:
        # If network is not connected, use weakly connected components
        all_paths_lengths = {}
        for component in nx.weakly_connected_components(network):
            subgraph = network.subgraph(component)
            paths = dict(nx.all_pairs_shortest_path_length(subgraph))
            all_paths_lengths.update(paths)
    
    # Calculate error propagation depth
    depths = []
    for agent_id, errors in agent_errors.items():
        error_count = np.sum(errors)
        if error_count > 0:
            # Find distance to other agents
            if agent_id in all_paths_lengths:
                neighbor_distances = all_paths_lengths[agent_id]
                avg_distance = np.mean(list(neighbor_distances.values())) if neighbor_distances else 0
                depths.append(avg_distance)
    
    if depths:
        mean_depth = np.mean(depths)
        max_depth = np.max(depths)
    else:
        mean_depth = 0.0
        max_depth = 0.0
    
    # Overall error rate
    all_preds = np.array(list(agent_predictions.values()))
    consensus_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, all_preds)
    error_rate = np.mean(consensus_preds != test_labels)
    
    return {
        'mean_depth': mean_depth,
        'max_depth': max_depth,
        'error_rate': error_rate
    }


def calculate_failed_node_centrality(agents: List, network: nx.DiGraph) -> Dict[str, float]:
    """
    Calculate centrality measures for failed nodes
    
    Higher centrality of failed nodes means greater impact on system
    
    Args:
        agents: List of agents
        network: Communication network
        
    Returns:
        Dictionary with centrality metrics
    """
    failed_nodes = [agent.agent_id for agent in agents if agent.failed]
    
    if not failed_nodes:
        return {
            'avg_degree_centrality': 0.0,
            'avg_betweenness_centrality': 0.0,
            'avg_closeness_centrality': 0.0,
            'failure_rate': 0.0
        }
    
    # Calculate centralities
    try:
        degree_centrality = nx.degree_centrality(network.to_undirected())
        betweenness_centrality = nx.betweenness_centrality(network)
        closeness_centrality = nx.closeness_centrality(network)
    except:
        # Fallback if calculation fails
        return {
            'avg_degree_centrality': 0.0,
            'avg_betweenness_centrality': 0.0,
            'avg_closeness_centrality': 0.0,
            'failure_rate': len(failed_nodes) / len(agents)
        }
    
    # Average centrality of failed nodes
    avg_degree = np.mean([degree_centrality.get(node, 0) for node in failed_nodes])
    avg_betweenness = np.mean([betweenness_centrality.get(node, 0) for node in failed_nodes])
    avg_closeness = np.mean([closeness_centrality.get(node, 0) for node in failed_nodes])
    
    return {
        'avg_degree_centrality': avg_degree,
        'avg_betweenness_centrality': avg_betweenness,
        'avg_closeness_centrality': avg_closeness,
        'failure_rate': len(failed_nodes) / len(agents)
    }


def evaluate_system(agents: List, network: nx.DiGraph,
                   test_data: np.ndarray, test_labels: np.ndarray,
                   accuracy_history: List[float] = None,
                   failure_history: List[int] = None) -> Dict[str, Any]:
    """
    Comprehensive system evaluation
    
    Args:
        agents: List of agents
        network: Communication network
        test_data: Test data (full test set, but agents only see their subset)
        test_labels: True labels (full test set)
        accuracy_history: History of accuracy values
        failure_history: History of failed node counts
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Collect predictions from each agent on their local test subset
    all_predictions = {}  # Map from test index to list of predictions
    all_labels = {}  # Map from test index to true label
    
    for agent in agents:
        if not agent.failed and hasattr(agent, 'local_test_data') and agent.local_test_data is not None:
            # Agent predicts on their local test subset
            pred = agent.predict(agent.local_test_data)
            
            # Store predictions with their original test indices
            for local_idx, test_idx in enumerate(agent.local_test_indices):
                if test_idx not in all_predictions:
                    all_predictions[test_idx] = []
                    all_labels[test_idx] = agent.local_test_labels[local_idx]
                all_predictions[test_idx].append(pred[local_idx])
    
    if all_predictions:
        # Consensus prediction (majority vote) for each test sample
        final_predictions = []
        final_labels = []
        
        for test_idx in sorted(all_predictions.keys()):
            preds = all_predictions[test_idx]
            if preds:
                # Majority vote
                consensus = np.bincount(preds, minlength=len(np.unique(test_labels))).argmax()
                final_predictions.append(consensus)
                final_labels.append(all_labels[test_idx])
        
        if final_predictions:
            final_predictions = np.array(final_predictions)
            final_labels = np.array(final_labels)
            accuracy = calculate_accuracy(final_predictions, final_labels)
        else:
            accuracy = 0.0
    else:
        accuracy = 0.0
    
    # Calculate other metrics
    if accuracy_history and failure_history:
        robustness = calculate_robustness(accuracy_history, failure_history, network)
    else:
        robustness = 1.0
    
    error_depth_metrics = calculate_error_depth(agents, network, test_data, test_labels)
    centrality_metrics = calculate_failed_node_centrality(agents, network)
    
    # Combine all metrics
    results = {
        'accuracy': accuracy,
        'robustness': robustness,
        'error_depth': error_depth_metrics,
        'failed_node_centrality': centrality_metrics,
        'n_agents': len(agents),
        'n_failed': sum(1 for a in agents if a.failed),
        'n_edges': network.number_of_edges()
    }
    
    return results

