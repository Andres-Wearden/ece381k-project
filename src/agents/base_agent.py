"""Base agent class for multi-agent systems"""

import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class for agents in the network"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int):
        self.agent_id = agent_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.failed = False
        self.message_queue = []
        self.received_messages = []
        self.local_data = None
        self.local_labels = None
        self.local_test_data = None
        self.local_test_labels = None
        self.local_test_indices = None
        self.prediction_history = []
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the local model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for sharing"""
        pass
    
    @abstractmethod
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model based on received messages"""
        pass
    
    def receive_message(self, message: Dict[str, Any], delay: int = 0):
        """Receive a message with optional delay"""
        self.message_queue.append({
            'message': message,
            'delay': delay,
            'timestamp': 0  # Will be set by simulation
        })
    
    def process_messages(self, current_time: int):
        """Process messages that are ready (delay expired)"""
        ready_messages = []
        remaining_messages = []
        
        for msg_item in self.message_queue:
            if current_time >= msg_item['timestamp'] + msg_item['delay']:
                ready_messages.append(msg_item['message'])
            else:
                remaining_messages.append(msg_item)
        
        self.message_queue = remaining_messages
        
        if ready_messages and not self.failed:
            self.update_from_messages(ready_messages)
            self.received_messages.extend(ready_messages)
    
    def set_local_data(self, X: np.ndarray, y: np.ndarray):
        """Set local training data"""
        self.local_data = X
        self.local_labels = y
    
    def set_local_test_data(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        """Set local test data for this agent"""
        self.local_test_data = X
        self.local_test_labels = y
        self.local_test_indices = indices
    
    def fail(self):
        """Simulate agent failure"""
        self.failed = True
    
    def recover(self):
        """Recover failed agent"""
        self.failed = False
    
    def reset(self):
        """Reset agent state"""
        self.message_queue = []
        self.received_messages = []
        self.prediction_history = []
        self.failed = False

