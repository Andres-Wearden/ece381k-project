"""Simulation engine for multi-agent systems"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Type
from tqdm import tqdm
import copy

from ..agents.base_agent import Agent
from .perturbations import Perturbation


class SimulationEngine:
    """
    Main simulation engine for multi-agent AI systems
    
    Handles message passing, agent updates, and perturbations
    """
    
    def __init__(self, agents: List[Agent], network: nx.DiGraph,
                 test_data: np.ndarray, test_labels: np.ndarray,
                 data_graph: Optional[nx.Graph] = None,
                 test_indices: Optional[np.ndarray] = None):
        """
        Initialize simulation
        
        Args:
            agents: List of agent instances
            network: Communication network (directed graph)
            test_data: Test data for evaluation
            test_labels: Test labels for evaluation
            data_graph: Graph structure from dataset (for GAT/GNN agents)
            test_indices: Original indices of test samples (for graph structure)
        """
        self.agents = agents
        self.network = network
        self.test_data = test_data
        self.test_labels = test_labels
        self.data_graph = data_graph
        self.test_indices = test_indices if test_indices is not None else np.arange(len(test_data))
        self.perturbations = []
        self.history = {
            'accuracy': [],
            'failed_nodes': [],
            'message_counts': [],
            'predictions': []
        }
        self.current_time = 0
    
    def add_perturbation(self, perturbation: Perturbation):
        """Add a perturbation to the simulation"""
        self.perturbations.append(perturbation)
    
    def distribute_data(self, train_data: np.ndarray, train_labels: np.ndarray,
                       distribution: str = 'equal', samples_per_agent: int = None,
                       train_indices: Optional[np.ndarray] = None):
        """
        Distribute training data among agents
        
        Args:
            train_data: Training features
            train_labels: Training labels
            distribution: Distribution strategy ('equal', 'random', 'biased', 'overlap')
            samples_per_agent: If specified, each agent gets this many samples (with overlap if needed)
            train_indices: Original indices of training samples (for graph structure)
        """
        n_agents = len(self.agents)
        n_samples = len(train_data)
        
        # Get node indices for graph structure
        if train_indices is None:
            train_indices = np.arange(n_samples)
        
        if samples_per_agent is not None:
            # Distribute with specified samples per agent (allows overlap)
            for i, agent in enumerate(self.agents):
                # Use different random seeds per agent for variety
                np.random.seed(42 + i)
                sample_indices = np.random.choice(n_samples, size=min(samples_per_agent, n_samples), replace=True)
                agent.set_local_data(train_data[sample_indices], train_labels[sample_indices])
                # Set graph structure for GAT/GNN agents
                if hasattr(agent, 'set_local_graph') and self.data_graph is not None:
                    node_indices = train_indices[sample_indices]
                    adj = self._get_subgraph_adjacency(node_indices)
                    agent.set_local_graph(node_indices, adj)
            np.random.seed(42)  # Reset seed
            return
        
        if distribution == 'equal':
            # Distribute data equally
            samples_per_agent = n_samples // n_agents
            for i, agent in enumerate(self.agents):
                start_idx = i * samples_per_agent
                end_idx = start_idx + samples_per_agent if i < n_agents - 1 else n_samples
                agent_indices = np.arange(start_idx, end_idx)
                agent.set_local_data(train_data[start_idx:end_idx], 
                                    train_labels[start_idx:end_idx])
                # Set graph structure for GAT/GNN agents
                if hasattr(agent, 'set_local_graph') and self.data_graph is not None:
                    node_indices = train_indices[agent_indices]
                    adj = self._get_subgraph_adjacency(node_indices)
                    agent.set_local_graph(node_indices, adj)
        
        elif distribution == 'random':
            # Randomly distribute data
            indices = np.random.permutation(n_samples)
            samples_per_agent = n_samples // n_agents
            for i, agent in enumerate(self.agents):
                start_idx = i * samples_per_agent
                end_idx = start_idx + samples_per_agent if i < n_agents - 1 else n_samples
                agent_indices = indices[start_idx:end_idx]
                agent.set_local_data(train_data[agent_indices], 
                                    train_labels[agent_indices])
                # Set graph structure for GAT/GNN agents
                if hasattr(agent, 'set_local_graph') and self.data_graph is not None:
                    node_indices = train_indices[agent_indices]
                    adj = self._get_subgraph_adjacency(node_indices)
                    agent.set_local_graph(node_indices, adj)
        
        elif distribution == 'biased':
            # Give some agents more data than others
            proportions = np.random.dirichlet(np.ones(n_agents) * 0.5)
            start_idx = 0
            for i, agent in enumerate(self.agents):
                n_agent_samples = int(proportions[i] * n_samples)
                end_idx = min(start_idx + n_agent_samples, n_samples)
                if i == n_agents - 1:
                    end_idx = n_samples
                agent_indices = np.arange(start_idx, end_idx)
                agent.set_local_data(train_data[start_idx:end_idx], 
                                    train_labels[start_idx:end_idx])
                # Set graph structure for GAT/GNN agents
                if hasattr(agent, 'set_local_graph') and self.data_graph is not None:
                    node_indices = train_indices[agent_indices]
                    adj = self._get_subgraph_adjacency(node_indices)
                    agent.set_local_graph(node_indices, adj)
                start_idx = end_idx
        
        elif distribution == 'overlap':
            # Each agent gets overlapping data (more data per agent)
            overlap_ratio = 0.3  # 30% overlap
            samples_per_agent = int(n_samples * (1 + overlap_ratio) / n_agents)
            for i, agent in enumerate(self.agents):
                start_idx = max(0, i * samples_per_agent - int(overlap_ratio * samples_per_agent))
                end_idx = min(start_idx + samples_per_agent, n_samples)
                agent_indices = np.arange(start_idx, end_idx)
                agent.set_local_data(train_data[start_idx:end_idx], 
                                    train_labels[start_idx:end_idx])
                # Set graph structure for GAT/GNN agents
                if hasattr(agent, 'set_local_graph') and self.data_graph is not None:
                    node_indices = train_indices[agent_indices]
                    adj = self._get_subgraph_adjacency(node_indices)
                    agent.set_local_graph(node_indices, adj)
    
    def _get_subgraph_adjacency(self, node_indices: np.ndarray) -> np.ndarray:
        """Get adjacency matrix for a subgraph of nodes"""
        if self.data_graph is None:
            # No graph structure, return identity
            return np.eye(len(node_indices))
        
        n_nodes = len(node_indices)
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Create mapping from original node indices to local indices
        node_to_local = {node: i for i, node in enumerate(node_indices)}
        
        # Build adjacency matrix for subgraph
        for i, node in enumerate(node_indices):
            if node in self.data_graph:
                # Get neighbors in the data graph
                neighbors = list(self.data_graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor in node_to_local:
                        j = node_to_local[neighbor]
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0  # Make symmetric
        
        # Add self-loops
        adj += np.eye(n_nodes)
        
        return adj
    
    def distribute_test_data(self, distribution: str = 'equal'):
        """Distribute test data among agents (each agent only sees their subset)"""
        n_agents = len(self.agents)
        n_test_samples = len(self.test_data)
        
        if distribution == 'equal':
            # Distribute test data equally
            samples_per_agent = n_test_samples // n_agents
            for i, agent in enumerate(self.agents):
                start_idx = i * samples_per_agent
                end_idx = start_idx + samples_per_agent if i < n_agents - 1 else n_test_samples
                test_indices = np.arange(start_idx, end_idx)
                agent.set_local_test_data(
                    self.test_data[start_idx:end_idx],
                    self.test_labels[start_idx:end_idx],
                    self.test_indices[test_indices]
                )
                
                # For graph-based agents, set up graph structure for their test subset
                if hasattr(agent, 'set_local_graph') and self.data_graph is not None:
                    # Get subgraph that includes both training and test nodes for this agent
                    train_node_indices = agent.local_graph['node_indices'] if agent.local_graph else np.array([])
                    test_node_indices = self.test_indices[test_indices]
                    
                    # Combine training and test nodes
                    all_node_indices = np.concatenate([train_node_indices, test_node_indices])
                    all_node_indices = np.unique(all_node_indices)
                    
                    # Get subgraph adjacency for combined nodes
                    adj = self._get_subgraph_adjacency(all_node_indices)
                    
                    # Store mapping for prediction
                    agent.test_graph = {
                        'node_indices': all_node_indices,
                        'adj_matrix': adj,
                        'test_node_local_indices': np.arange(len(train_node_indices), len(all_node_indices))
                    }
    
    def set_full_graph_for_agents(self):
        """Set full graph adjacency matrix for all graph-based agents"""
        if self.data_graph is None:
            return
        
        # Build full adjacency matrix (for reference, but agents use local subgraphs)
        n_nodes = len(self.data_graph)
        full_adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        for node in self.data_graph.nodes():
            neighbors = list(self.data_graph.neighbors(node))
            for neighbor in neighbors:
                full_adj[node, neighbor] = 1.0
                full_adj[neighbor, node] = 1.0  # Make symmetric
        
        # Add self-loops
        full_adj += np.eye(n_nodes)
        
        # Store full graph for agents that need it (for extracting subgraphs)
        for agent in self.agents:
            if hasattr(agent, 'set_full_graph'):
                agent.set_full_graph(full_adj)
    
    def train_agents(self):
        """Train all agents on their local data"""
        for agent in self.agents:
            if agent.local_data is not None and not agent.failed:
                agent.train(agent.local_data, agent.local_labels)
    
    def communicate(self):
        """Execute one round of communication between agents"""
        # Collect messages to send
        messages_to_send = []
        
        for u, v in self.network.edges():
            if not self.agents[u].failed:
                # Agent u sends its parameters to agent v
                params = self.agents[u].get_model_params()
                weight = self.network[u][v].get('weight', 1.0)
                delay = self.network[u][v].get('delay', 0)
                
                message = {
                    'params': params,
                    'weight': weight,
                    'sender': u,
                    'receiver': v
                }
                
                messages_to_send.append((v, message, delay))
        
        # Send messages
        for receiver, message, delay in messages_to_send:
            self.agents[receiver].receive_message(message, delay)
            # Set timestamp
            if self.agents[receiver].message_queue:
                self.agents[receiver].message_queue[-1]['timestamp'] = self.current_time
    
    def update_agents(self):
        """Update agents based on received messages"""
        for agent in self.agents:
            agent.process_messages(self.current_time)
    
    def evaluate(self) -> float:
        """
        Evaluate current system performance
        
        Returns:
            Overall accuracy
        """
        # Collect predictions from each agent on their local test subset
        all_predictions = {}  # Map from test index to list of (prediction, weight)
        all_labels = {}  # Map from test index to true label
        
        for agent in self.agents:
            if not agent.failed and agent.local_test_data is not None:
                # Agent predicts on their local test subset
                pred = agent.predict(agent.local_test_data)
                
                # Weight by node centrality or degree
                degree = self.network.in_degree(agent.agent_id) + self.network.out_degree(agent.agent_id)
                weight = degree + 1  # Add 1 to avoid zero weight
                
                # Store predictions with their original test indices
                for local_idx, test_idx in enumerate(agent.local_test_indices):
                    if test_idx not in all_predictions:
                        all_predictions[test_idx] = []
                        all_labels[test_idx] = agent.local_test_labels[local_idx]
                    all_predictions[test_idx].append((pred[local_idx], weight))
        
        if not all_predictions:
            return 0.0
        
        # Aggregate predictions for each test sample
        final_predictions = []
        final_labels = []
        
        for test_idx in sorted(all_predictions.keys()):
            preds_with_weights = all_predictions[test_idx]
            if preds_with_weights:
                # Weighted voting
                preds, weights = zip(*preds_with_weights)
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                votes = np.bincount(preds, weights=weights, minlength=self.test_labels.max() + 1)
                final_predictions.append(np.argmax(votes))
                final_labels.append(all_labels[test_idx])
        
        if not final_predictions:
            return 0.0
        
        final_predictions = np.array(final_predictions)
        final_labels = np.array(final_labels)
        accuracy = np.mean(final_predictions == final_labels)
        
        return accuracy
    
    def run(self, n_rounds: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Run simulation for specified number of rounds
        
        Args:
            n_rounds: Number of communication rounds
            verbose: Show progress bar
            
        Returns:
            Simulation results
        """
        # Set full graph for agents (for extracting subgraphs)
        self.set_full_graph_for_agents()
        
        # Distribute test data to agents (each agent only sees their subset)
        self.distribute_test_data(distribution='equal')
        
        # Initial training
        self.train_agents()
        
        # Evaluate initial state
        initial_accuracy = self.evaluate()
        self.history['accuracy'].append(initial_accuracy)
        self.history['failed_nodes'].append(sum(1 for a in self.agents if a.failed))
        self.history['message_counts'].append(0)
        
        # Run simulation rounds
        iterator = tqdm(range(n_rounds), desc="Simulation") if verbose else range(n_rounds)
        
        for round_num in iterator:
            self.current_time = round_num + 1
            
            # Apply perturbations
            for perturbation in self.perturbations:
                perturbation.apply(self.agents, self.network, self.current_time)
            
            # Communication phase
            self.communicate()
            
            # Update phase
            self.update_agents()
            
            # Evaluation
            accuracy = self.evaluate()
            failed_count = sum(1 for a in self.agents if a.failed)
            message_count = sum(len(a.received_messages) for a in self.agents)
            
            self.history['accuracy'].append(accuracy)
            self.history['failed_nodes'].append(failed_count)
            self.history['message_counts'].append(message_count)
            
            if verbose:
                iterator.set_postfix({
                    'acc': f'{accuracy:.3f}',
                    'failed': failed_count
                })
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results"""
        return {
            'history': self.history,
            'final_accuracy': self.history['accuracy'][-1],
            'average_accuracy': np.mean(self.history['accuracy']),
            'accuracy_std': np.std(self.history['accuracy']),
            'max_failed_nodes': max(self.history['failed_nodes']),
            'total_messages': self.history['message_counts'][-1],
            'network_info': {
                'n_agents': len(self.agents),
                'n_edges': self.network.number_of_edges(),
                'density': nx.density(self.network)
            }
        }
    
    def reset(self):
        """Reset simulation state"""
        self.current_time = 0
        self.history = {
            'accuracy': [],
            'failed_nodes': [],
            'message_counts': [],
            'predictions': []
        }
        for agent in self.agents:
            agent.reset()

