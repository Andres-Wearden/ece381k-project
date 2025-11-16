"""Evaluation metrics for multi-agent benchmarking"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any
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
                         failure_history: List[int]) -> float:
    """
    Calculate system robustness as stability under perturbations
    
    Robustness = (1 - accuracy_variance) * (1 - failure_impact)
    
    Args:
        accuracy_history: List of accuracy values over time
        failure_history: List of failed node counts over time
        
    Returns:
        Robustness score (0-1, higher is better)
    """
    if len(accuracy_history) < 2:
        return 1.0
    
    # Measure accuracy stability
    accuracy_var = np.var(accuracy_history)
    stability = 1.0 / (1.0 + accuracy_var)
    
    # Measure failure impact
    if max(failure_history) > 0:
        # Find accuracy drop during failures
        failure_indices = [i for i, f in enumerate(failure_history) if f > 0]
        if failure_indices:
            accuracies_with_failures = [accuracy_history[i] for i in failure_indices]
            accuracies_without_failures = [accuracy_history[i] for i, f in enumerate(failure_history) if f == 0]
            
            if accuracies_without_failures:
                avg_normal = np.mean(accuracies_without_failures)
                avg_failure = np.mean(accuracies_with_failures)
                failure_impact = max(0, avg_normal - avg_failure)
            else:
                failure_impact = 0.0
        else:
            failure_impact = 0.0
    else:
        failure_impact = 0.0
    
    # Combined robustness score
    robustness = stability * (1.0 - failure_impact)
    
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
        robustness = calculate_robustness(accuracy_history, failure_history)
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

