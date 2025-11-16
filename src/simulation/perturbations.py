"""Perturbation classes for simulation"""

import numpy as np
import networkx as nx
from typing import List
from abc import ABC, abstractmethod


class Perturbation(ABC):
    """Base class for perturbations"""
    
    @abstractmethod
    def apply(self, agents: List, network: nx.DiGraph, time_step: int):
        """Apply perturbation to the system"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of perturbation"""
        pass


class NodeFailure(Perturbation):
    """Simulate node/agent failures"""
    
    def __init__(self, failure_prob: float = 0.1, failure_duration: int = 5,
                 start_time: int = 5, end_time: int = None):
        self.failure_prob = failure_prob
        self.failure_duration = failure_duration
        self.start_time = start_time
        self.end_time = end_time
        self.failed_agents = {}  # agent_id -> recovery_time
    
    def apply(self, agents: List, network: nx.DiGraph, time_step: int):
        """Apply node failures"""
        if time_step < self.start_time:
            return
        
        if self.end_time is not None and time_step > self.end_time:
            return
        
        # Recover agents whose duration has expired
        to_recover = []
        for agent_id, recovery_time in self.failed_agents.items():
            if time_step >= recovery_time:
                agents[agent_id].recover()
                to_recover.append(agent_id)
        
        for agent_id in to_recover:
            del self.failed_agents[agent_id]
        
        # Randomly fail agents
        for agent in agents:
            if agent.agent_id not in self.failed_agents:
                if np.random.rand() < self.failure_prob:
                    agent.fail()
                    self.failed_agents[agent.agent_id] = time_step + self.failure_duration
    
    def get_description(self) -> str:
        return f"NodeFailure(prob={self.failure_prob}, duration={self.failure_duration})"


class DelayPerturbation(Perturbation):
    """Add or modify communication delays"""
    
    def __init__(self, delay_increase: int = 2, affected_prob: float = 0.3,
                 start_time: int = 5, end_time: int = None):
        self.delay_increase = delay_increase
        self.affected_prob = affected_prob
        self.start_time = start_time
        self.end_time = end_time
        self.original_delays = {}
    
    def apply(self, agents: List, network: nx.DiGraph, time_step: int):
        """Apply delay perturbations"""
        if time_step < self.start_time:
            return
        
        # Apply delays
        if self.end_time is None or time_step <= self.end_time:
            for u, v in network.edges():
                edge_key = (u, v)
                if edge_key not in self.original_delays:
                    self.original_delays[edge_key] = network[u][v].get('delay', 0)
                
                if np.random.rand() < self.affected_prob:
                    network[u][v]['delay'] = self.original_delays[edge_key] + self.delay_increase
        else:
            # Restore original delays
            for (u, v), original_delay in self.original_delays.items():
                if network.has_edge(u, v):
                    network[u][v]['delay'] = original_delay
    
    def get_description(self) -> str:
        return f"DelayPerturbation(increase={self.delay_increase}, prob={self.affected_prob})"


class WeightPerturbation(Perturbation):
    """Modify edge weights (e.g., communication quality degradation)"""
    
    def __init__(self, weight_factor: float = 0.5, affected_prob: float = 0.3,
                 start_time: int = 5, end_time: int = None):
        self.weight_factor = weight_factor
        self.affected_prob = affected_prob
        self.start_time = start_time
        self.end_time = end_time
        self.original_weights = {}
    
    def apply(self, agents: List, network: nx.DiGraph, time_step: int):
        """Apply weight perturbations"""
        if time_step < self.start_time:
            return
        
        if self.end_time is None or time_step <= self.end_time:
            for u, v in network.edges():
                edge_key = (u, v)
                if edge_key not in self.original_weights:
                    self.original_weights[edge_key] = network[u][v].get('weight', 1.0)
                
                if np.random.rand() < self.affected_prob:
                    network[u][v]['weight'] = self.original_weights[edge_key] * self.weight_factor
        else:
            # Restore original weights
            for (u, v), original_weight in self.original_weights.items():
                if network.has_edge(u, v):
                    network[u][v]['weight'] = original_weight
    
    def get_description(self) -> str:
        return f"WeightPerturbation(factor={self.weight_factor}, prob={self.affected_prob})"

