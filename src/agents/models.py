"""Agent model implementations"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from .base_agent import Agent
from .aggregation import AggregationStrategy, AverageAggregation, get_aggregation_strategy

# Suppress sklearn FutureWarnings about deprecated parameters
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


class LogisticAgent(Agent):
    """Agent with logistic regression model"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int, 
                 C: float = 1.0, max_iter: int = 1000,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.C = C
        self.max_iter = max_iter
        # Note: multi_class parameter removed as it's deprecated in sklearn 1.5+
        # sklearn will automatically use 'multinomial' for multiclass problems
        self.model = LogisticRegression(
            C=C, 
            max_iter=max_iter, 
            random_state=42,
            solver='lbfgs'
        )
        self.trained = False
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train logistic regression model"""
        if self.failed:
            return
        
        if len(X) > 0 and len(np.unique(y)) > 1:
            try:
                self.model.fit(X, y)
                self.trained = True
            except:
                # If training fails, use simple default
                self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.failed or not self.trained:
            # Return random predictions if failed or untrained
            return np.zeros(len(X), dtype=int)
        
        try:
            return self.model.predict(X)
        except:
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.trained:
            return {'coef': None, 'intercept': None, 'trained': False}
        
        return {
            'coef': self.model.coef_.copy() if hasattr(self.model, 'coef_') else None,
            'intercept': self.model.intercept_.copy() if hasattr(self.model, 'intercept_') else None,
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        if self.failed or not messages:
            return
        
        # Prepare own parameters
        own_params = None
        if self.trained:
            own_params = {
                'coef': self.model.coef_.copy() if hasattr(self.model, 'coef_') else None,
                'intercept': self.model.intercept_.copy() if hasattr(self.model, 'intercept_') else None,
                'trained': self.trained
            }
        
        # Use aggregation strategy
        aggregated = self.aggregation_strategy.aggregate(messages, own_params)
        
        if aggregated and aggregated.get('trained', False):
            if 'coef' in aggregated and aggregated['coef'] is not None:
                self.model.coef_ = aggregated['coef']
                self.model.intercept_ = aggregated['intercept']
                self.trained = True


class LinearAgent(Agent):
    """Agent with linear regression model"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int, 
                 alpha: float = 1.0,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.trained = False
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train linear model"""
        if self.failed:
            return
        
        if len(X) > 0:
            try:
                self.model.fit(X, y)
                self.trained = True
            except:
                self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.failed or not self.trained:
            return np.zeros(len(X), dtype=int)
        
        try:
            preds = self.model.predict(X)
            # Round and clip for classification
            return np.clip(np.round(preds), 0, self.n_classes - 1).astype(int)
        except:
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.trained:
            return {'coef': None, 'intercept': None, 'trained': False}
        
        return {
            'coef': self.model.coef_.copy() if hasattr(self.model, 'coef_') else None,
            'intercept': self.model.intercept_.copy() if hasattr(self.model, 'intercept_') else None,
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        if self.failed or not messages:
            return
        
        # Prepare own parameters
        own_params = None
        if self.trained:
            own_params = {
                'coef': self.model.coef_.copy() if hasattr(self.model, 'coef_') else None,
                'intercept': self.model.intercept_.copy() if hasattr(self.model, 'intercept_') else None,
                'trained': self.trained
            }
        
        # Use aggregation strategy
        aggregated = self.aggregation_strategy.aggregate(messages, own_params)
        
        if aggregated and aggregated.get('trained', False):
            if 'coef' in aggregated and aggregated['coef'] is not None:
                self.model.coef_ = aggregated['coef']
                self.model.intercept_ = aggregated['intercept']
                self.trained = True


class SimpleNN(nn.Module):
    """Neural network for agents with configurable depth"""
    
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 32, n_layers: int = 10):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        
        # Build layers dynamically
        self.layers = nn.ModuleList()
        
        # First layer: input → hidden
        self.layers.append(nn.Linear(n_features, hidden_dim))
        
        # Hidden layers: hidden → hidden (n_layers - 2 of these)
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Final layer: hidden → output
        self.layers.append(nn.Linear(hidden_dim, n_classes))
    
    def forward(self, x):
        # Apply all layers except the last with ReLU
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.relu(x)
        
        # Final layer (no activation - raw logits)
        x = self.layers[-1](x)
        return x


class NeuralAgent(Agent):
    """Agent with neural network"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int,
                 hidden_dim: int = 32, lr: float = 0.01, epochs: int = 50,
                 n_layers: int = 10,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.n_layers = n_layers
        self.model = SimpleNN(n_features, n_classes, hidden_dim, n_layers=n_layers)
        self.trained = False
        self.device = torch.device('cpu')
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train neural network"""
        if self.failed:
            return
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            return
        
        try:
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            self.trained = True
        except:
            self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.failed or not self.trained:
            return np.zeros(len(X), dtype=int)
        
        try:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        except:
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.trained:
            return {'state_dict': None, 'trained': False}
        
        return {
            'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        if self.failed or not messages:
            return
        
        # Prepare own parameters
        own_params = None
        if self.trained:
            own_params = {
                'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                'trained': self.trained
            }
        
        # Use aggregation strategy
        aggregated = self.aggregation_strategy.aggregate(messages, own_params)
        
        if aggregated and aggregated.get('trained', False):
            if 'state_dict' in aggregated and aggregated['state_dict'] is not None:
                self.model.load_state_dict(aggregated['state_dict'])
                self.trained = True


class GraphSAGELayer(nn.Module):
    """GraphSAGE layer with neighbor sampling and aggregation"""
    
    def __init__(self, in_features: int, out_features: int, aggregator: str = 'mean', num_samples: int = 10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        self.num_samples = num_samples
        
        # Linear transformation for self features
        self.self_linear = nn.Linear(in_features, out_features)
        
        # Linear transformation for neighbor features
        self.neighbor_linear = nn.Linear(in_features, out_features)
        
        if aggregator == 'lstm':
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)
        
    def aggregate(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor features - DEPRECATED: now done inline"""
        # This method is kept for compatibility but aggregation is now done inline
        if neighbor_features.size(0) == 0:
            return torch.zeros(self.in_features, device=neighbor_features.device)
        
        if self.aggregator == 'mean':
            return torch.mean(neighbor_features, dim=0)
        elif self.aggregator == 'max':
            return torch.max(neighbor_features, dim=0)[0]
        elif self.aggregator == 'lstm':
            # LSTM needs batch dimension
            lstm_input = neighbor_features.unsqueeze(0)  # [1, num_neighbors, in_features]
            lstm_out, _ = self.lstm(lstm_input)
            return lstm_out[0, -1, :]  # [in_features]
        else:
            return torch.mean(neighbor_features, dim=0)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Output features [N, out_features]
        """
        N = x.size(0)
        
        # Transform self features
        self_features = self.self_linear(x)  # [N, out_features]
        
        # Aggregate neighbor features for each node
        aggregated_neighbors = []
        for i in range(N):
            neighbors = torch.nonzero(adj[i] > 0, as_tuple=False).squeeze(-1)
            if len(neighbors) > 0:
                # Sample neighbors if needed
                if len(neighbors) > self.num_samples:
                    sampled_idx = torch.randperm(len(neighbors), device=x.device)[:self.num_samples]
                    neighbor_feat = x[neighbors[sampled_idx]]  # [num_samples, in_features]
                else:
                    neighbor_feat = x[neighbors]  # [num_neighbors, in_features]
                
                # Aggregate neighbor features
                if neighbor_feat.size(0) > 0:
                    # Aggregate over neighbors: [num_neighbors, in_features] -> [in_features]
                    if self.aggregator == 'mean':
                        aggregated = torch.mean(neighbor_feat, dim=0)  # [in_features]
                    elif self.aggregator == 'max':
                        aggregated = torch.max(neighbor_feat, dim=0)[0]  # [in_features]
                    elif self.aggregator == 'lstm':
                        # LSTM needs batch dimension
                        lstm_input = neighbor_feat.unsqueeze(0)  # [1, num_neighbors, in_features]
                        lstm_out, _ = self.lstm(lstm_input)
                        aggregated = lstm_out[0, -1, :]  # [in_features]
                    else:
                        aggregated = torch.mean(neighbor_feat, dim=0)  # [in_features]
                else:
                    aggregated = torch.zeros(self.in_features, device=x.device)
            else:
                # No neighbors, use zero
                aggregated = torch.zeros(self.in_features, device=x.device)
            
            # Transform aggregated neighbors
            aggregated_transformed = self.neighbor_linear(aggregated.unsqueeze(0))  # [1, out_features]
            aggregated_neighbors.append(aggregated_transformed.squeeze(0))
        
        aggregated_neighbors = torch.stack(aggregated_neighbors)  # [N, out_features]
        
        # Combine self and neighbor features
        output = self_features + aggregated_neighbors
        # Remove L2 normalization - it can hurt training
        # output = F.normalize(output, p=2, dim=1)  # L2 normalize
        
        return output


class GraphSAGEModel(nn.Module):
    """GraphSAGE Network"""
    
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, aggregator: str = 'mean', num_samples: int = 10):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GraphSAGELayer(n_features, hidden_dim, aggregator, num_samples))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator, num_samples))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, n_classes))
        else:
            self.layers.append(nn.Linear(n_features, n_classes))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers[:-1]):
            if isinstance(layer, GraphSAGELayer):
                x = layer(x, adj)
                x = self.activation(x)
                x = self.dropout(x)
            else:
                x = layer(x)
                x = self.activation(x)
        
        # Final layer
        x = self.layers[-1](x)
        return x


class GraphSAGEAgent(Agent):
    """Agent with GraphSAGE model"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 aggregator: str = 'mean', num_samples: int = 10,
                 lr: float = 0.01, epochs: int = 50, dropout: float = 0.0,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aggregator = aggregator
        self.num_samples = num_samples
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.model = GraphSAGEModel(n_features, n_classes, hidden_dim, num_layers, aggregator, num_samples)
        self.trained = False
        self.device = torch.device('cpu')
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
        self.local_graph = None
        self.full_graph = None  # Store full graph for test predictions
    
    def set_local_graph(self, node_indices: np.ndarray, adj_matrix: np.ndarray):
        """Set local subgraph for this agent"""
        self.local_graph = {
            'node_indices': node_indices,
            'adj_matrix': adj_matrix
        }
    
    def set_full_graph(self, full_graph_adj: np.ndarray):
        """Set full graph adjacency for test predictions"""
        self.full_graph = full_graph_adj
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train GraphSAGE model"""
        if self.failed:
            return
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            return
        
        if self.local_graph is None:
            adj = np.eye(len(X))
            self.set_local_graph(np.arange(len(X)), adj)
        
        try:
            adj = self.local_graph['adj_matrix']
            adj = adj + adj.T
            adj = adj + np.eye(len(adj))
            adj = (adj > 0).astype(np.float32)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            adj_tensor = torch.FloatTensor(adj).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor, adj_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            self.trained = True
        except Exception as e:
            print(f"GraphSAGE training error for agent {self.agent_id}: {e}")
            self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.failed or not self.trained:
            return np.zeros(len(X), dtype=int)
        
        try:
            # Check if adjacency matrix size matches input size
            if self.local_graph is None:
                adj = np.eye(len(X))
            else:
                adj = self.local_graph['adj_matrix']
                # If adjacency size doesn't match input size, use identity (no graph structure for test)
                if adj.shape[0] != len(X):
                    # For test data, use identity adjacency (no graph structure)
                    adj = np.eye(len(X))
                else:
                    adj = adj + adj.T
                    adj = adj + np.eye(len(adj))
                    adj = (adj > 0).astype(np.float32)
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                adj_tensor = torch.FloatTensor(adj).to(self.device)
                outputs = self.model(X_tensor, adj_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        except Exception as e:
            print(f"GIN prediction error for agent {self.agent_id}: {e}")
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.trained:
            return {'state_dict': None, 'trained': False}
        
        return {
            'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        if self.failed or not messages:
            return
        
        own_params = None
        if self.trained:
            own_params = {
                'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                'trained': self.trained
            }
        
        aggregated = self.aggregation_strategy.aggregate(messages, own_params)
        
        if aggregated and aggregated.get('trained', False):
            if 'state_dict' in aggregated and aggregated['state_dict'] is not None:
                self.model.load_state_dict(aggregated['state_dict'])
                self.trained = True


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer"""
    
    def __init__(self, in_features: int, out_features: int, eps: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = nn.Parameter(torch.tensor(eps))
        
        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Output features [N, out_features]
        """
        # Aggregate neighbors: sum of neighbor features
        neighbor_sum = torch.matmul(adj, x)  # [N, in_features]
        
        # GIN: (1 + eps) * h + sum(neighbors)
        aggregated = (1 + self.eps) * x + neighbor_sum  # [N, in_features]
        
        # Apply MLP
        output = self.mlp(aggregated)  # [N, out_features]
        
        return output


class GINModel(nn.Module):
    """Graph Isomorphism Network"""
    
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, eps: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GINLayer(n_features, hidden_dim, eps))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GINLayer(hidden_dim, hidden_dim, eps))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, n_classes))
        else:
            self.layers.append(nn.Linear(n_features, n_classes))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
        
        # Final layer
        x = self.layers[-1](x)
        return x


class GINAgent(Agent):
    """Agent with Graph Isomorphism Network"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int,
                 hidden_dim: int = 64, num_layers: int = 2, eps: float = 0.0,
                 lr: float = 0.01, epochs: int = 50,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eps = eps
        self.lr = lr
        self.epochs = epochs
        self.model = GINModel(n_features, n_classes, hidden_dim, num_layers, eps)
        self.trained = False
        self.device = torch.device('cpu')
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
        self.local_graph = None
        self.full_graph = None  # Store full graph for test predictions
    
    def set_local_graph(self, node_indices: np.ndarray, adj_matrix: np.ndarray):
        """Set local subgraph for this agent"""
        self.local_graph = {
            'node_indices': node_indices,
            'adj_matrix': adj_matrix
        }
    
    def set_full_graph(self, full_graph_adj: np.ndarray):
        """Set full graph adjacency for test predictions"""
        self.full_graph = full_graph_adj
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train GIN model"""
        if self.failed:
            return
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            return
        
        if self.local_graph is None:
            adj = np.eye(len(X))
            self.set_local_graph(np.arange(len(X)), adj)
        
        try:
            adj = self.local_graph['adj_matrix']
            adj = adj + adj.T
            adj = adj + np.eye(len(adj))
            adj = (adj > 0).astype(np.float32)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            adj_tensor = torch.FloatTensor(adj).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor, adj_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            self.trained = True
        except Exception as e:
            print(f"GIN training error for agent {self.agent_id}: {e}")
            self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on local test data"""
        if self.failed or not self.trained:
            return np.zeros(len(X), dtype=int)
        
        try:
            # Use test graph if available (includes both training and test nodes)
            if hasattr(self, 'test_graph') and self.test_graph is not None:
                # Extract subgraph for test nodes only
                test_local_indices = self.test_graph['test_node_local_indices']
                full_adj = self.test_graph['adj_matrix']
                
                # Get adjacency for test nodes (includes connections to training nodes)
                adj = full_adj[np.ix_(test_local_indices, test_local_indices)]
                adj = adj + adj.T
                adj = adj + np.eye(len(adj))
                adj = (adj > 0).astype(np.float32)
            elif self.local_graph is not None and self.local_graph['adj_matrix'].shape[0] == len(X):
                # Use local graph if size matches
                adj = self.local_graph['adj_matrix']
                adj = adj + adj.T
                adj = adj + np.eye(len(adj))
                adj = (adj > 0).astype(np.float32)
            else:
                # Fallback to identity (no graph structure)
                adj = np.eye(len(X))
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                adj_tensor = torch.FloatTensor(adj).to(self.device)
                outputs = self.model(X_tensor, adj_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        except Exception as e:
            print(f"GIN prediction error for agent {self.agent_id}: {e}")
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.trained:
            return {'state_dict': None, 'trained': False}
        
        return {
            'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        if self.failed or not messages:
            return
        
        own_params = None
        if self.trained:
            own_params = {
                'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                'trained': self.trained
            }
        
        aggregated = self.aggregation_strategy.aggregate(messages, own_params)
        
        if aggregated and aggregated.get('trained', False):
            if 'state_dict' in aggregated and aggregated['state_dict'] is not None:
                self.model.load_state_dict(aggregated['state_dict'])
                self.trained = True


class RandomForestAgent(Agent):
    """Agent with Random Forest model"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int,
                 n_estimators: int = 100, max_depth: int = 20,
                 min_samples_split: int = 5,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42 + agent_id,  # Different seed per agent
            n_jobs=1
        )
        self.trained = False
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest model"""
        if self.failed:
            return
        
        if len(X) > 0 and len(np.unique(y)) > 1:
            try:
                self.model.fit(X, y)
                self.trained = True
            except Exception as e:
                print(f"Random Forest training error for agent {self.agent_id}: {e}")
                self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.failed or not self.trained:
            return np.zeros(len(X), dtype=int)
        
        try:
            return self.model.predict(X)
        except:
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for sharing"""
        # Random Forest doesn't have simple parameters to share
        # We'll use voting-based aggregation instead
        if not self.trained:
            return {'model': None, 'trained': False}
        
        # For Random Forest, we can't easily share tree structures
        # Instead, we'll share the model for voting
        return {
            'model': self.model,  # Share the whole model (for voting)
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        # Random Forest aggregation: Use voting from multiple models
        if self.failed or not messages:
            return
        
        # Collect predictions from all models (including self)
        valid_models = []
        if self.trained:
            valid_models.append(self.model)
        
        for msg in messages:
            params = msg.get('params', {})
            if params.get('trained', False) and params.get('model') is not None:
                valid_models.append(params['model'])
        
        # Store ensemble for voting (we'll use the first model and update predictions via voting)
        if len(valid_models) > 0:
            # For simplicity, use the first model but note we have an ensemble
            # In practice, we'd implement proper voting during prediction
            self.model = valid_models[0]
            self.trained = True


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for GAT"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 1, dropout: float = 0.0, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Each head has out_features dimensions
        self.head_dim = out_features
        
        # W transforms to out_features * num_heads if concat, else out_features
        if concat:
            self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        else:
            self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N] (sparse or dense)
        Returns:
            Output features [N, out_features * num_heads] if concat else [N, out_features]
        """
        N = x.size(0)
        h = self.W(x)  # [N, out_features * num_heads]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # Compute attention scores: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Expand h for all pairs: h_i for all j, and h_j for all i
        h_i = h.unsqueeze(1).expand(N, N, self.num_heads, self.head_dim)  # [N, N, num_heads, head_dim]
        h_j = h.unsqueeze(0).expand(N, N, self.num_heads, self.head_dim)  # [N, N, num_heads, head_dim]
        
        # Concatenate and compute attention
        a_input = torch.cat([h_i, h_j], dim=3)  # [N, N, num_heads, 2*head_dim]
        # Compute attention scores: e_ij = a^T [h_i || h_j]
        # Reshape for matrix multiplication: [N*N*num_heads, 2*head_dim] @ [2*head_dim, 1] = [N*N*num_heads, 1]
        a_input_flat = a_input.view(-1, 2 * self.head_dim)  # [N*N*num_heads, 2*head_dim]
        e_flat = torch.matmul(a_input_flat, self.a).squeeze(-1)  # [N*N*num_heads]
        e = e_flat.view(N, N, self.num_heads)  # [N, N, num_heads]
        e = self.leaky_relu(e)  # [N, N, num_heads]
        
        # Apply adjacency mask - expand adj to match e's dimensions
        # adj is [N, N], we need [N, N, num_heads]
        adj_expanded = adj.unsqueeze(2)  # [N, N, 1]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # [N, N, num_heads]
        attention = self.dropout(attention)
        
        # Aggregate: h'_i = sum_j alpha_ij * h_j
        # attention: [N, N, num_heads], h: [N, num_heads, head_dim]
        # We need to compute: for each head, attention[:, :, head] @ h[:, head, :]
        h_prime_list = []
        for head in range(self.num_heads):
            # attention[:, :, head]: [N, N], h[:, head, :]: [N, head_dim]
            # We want: attention[:, :, head] @ h[:, head, :] -> [N, head_dim]
            att_head = attention[:, :, head]  # [N, N]
            h_head = h[:, head, :]  # [N, head_dim]
            h_prime_head = torch.matmul(att_head, h_head)  # [N, head_dim]
            h_prime_list.append(h_prime_head)
        
        # Stack heads: [num_heads, N, head_dim] -> [N, num_heads, head_dim]
        h_prime = torch.stack(h_prime_list, dim=1)  # [N, num_heads, head_dim]
        
        if self.concat:
            # Concatenate heads: [N, num_heads * head_dim]
            h_prime = h_prime.view(N, self.num_heads * self.head_dim)  # [N, out_features * num_heads]
        else:
            # Average heads: [N, head_dim]
            h_prime = torch.mean(h_prime, dim=1)  # [N, out_features]
        
        return h_prime


class GATModel(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 64, 
                 num_heads: int = 2, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer: input → output with averaging (not concatenation)
            self.layers.append(GraphAttentionLayer(n_features, n_classes, 1, dropout, concat=False))
        else:
            # First layer: n_features → hidden_dim with num_heads (concatenated)
            # Output: hidden_dim * num_heads
            self.layers.append(GraphAttentionLayer(n_features, hidden_dim, num_heads, dropout, concat=True))
            
            # Hidden layers: hidden_dim * num_heads → hidden_dim with num_heads (concatenated)
            # Output: hidden_dim * num_heads
            for _ in range(num_layers - 2):
                self.layers.append(GraphAttentionLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout, concat=True))
            
            # Output layer: hidden_dim * num_heads → n_classes with 1 head (averaged)
            # Output: n_classes
            self.layers.append(GraphAttentionLayer(hidden_dim * num_heads, n_classes, 1, dropout, concat=False))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final layer
        if isinstance(self.layers[-1], GraphAttentionLayer):
            x = self.layers[-1](x, adj)
        else:
            x = self.layers[-1](x)
        
        return x


class GATAgent(Agent):
    """Agent with Graph Attention Network"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int,
                 hidden_dim: int = 64, num_heads: int = 2, num_layers: int = 2,
                 lr: float = 0.01, epochs: int = 50, dropout: float = 0.0,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.model = GATModel(n_features, n_classes, hidden_dim, num_heads, num_layers, dropout)
        self.trained = False
        self.device = torch.device('cpu')
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
        self.local_graph = None  # Will store local subgraph
        self.full_graph = None  # Store full graph for test predictions
    
    def set_local_graph(self, node_indices: np.ndarray, adj_matrix: np.ndarray):
        """Set local subgraph for this agent"""
        self.local_graph = {
            'node_indices': node_indices,
            'adj_matrix': adj_matrix
        }
    
    def set_full_graph(self, full_graph_adj: np.ndarray):
        """Set full graph adjacency for test predictions"""
        self.full_graph = full_graph_adj
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train GAT model"""
        if self.failed:
            return
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            return
        
        if self.local_graph is None:
            # Fallback: create identity adjacency (no graph structure)
            adj = np.eye(len(X))
            self.set_local_graph(np.arange(len(X)), adj)
        
        try:
            adj = self.local_graph['adj_matrix']
            # Ensure adjacency is symmetric and has self-loops
            adj = adj + adj.T
            adj = adj + np.eye(len(adj))
            adj = (adj > 0).astype(np.float32)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            adj_tensor = torch.FloatTensor(adj).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor, adj_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            self.trained = True
        except Exception as e:
            print(f"GAT training error for agent {self.agent_id}: {e}")
            self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on local test data"""
        if self.failed or not self.trained:
            return np.zeros(len(X), dtype=int)
        
        try:
            # For test data, we typically don't have the graph structure
            # Use identity matrix (each node is independent) for test predictions
            adj = np.eye(len(X))
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                adj_tensor = torch.FloatTensor(adj).to(self.device)
                outputs = self.model(X_tensor, adj_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        except Exception as e:
            print(f"GAT prediction error for agent {self.agent_id}: {e}")
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.trained:
            return {'state_dict': None, 'trained': False}
        
        return {
            'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        if self.failed or not messages:
            return
        
        own_params = None
        if self.trained:
            own_params = {
                'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                'trained': self.trained
            }
        
        aggregated = self.aggregation_strategy.aggregate(messages, own_params)
        
        if aggregated and aggregated.get('trained', False):
            if 'state_dict' in aggregated and aggregated['state_dict'] is not None:
                self.model.load_state_dict(aggregated['state_dict'])
                self.trained = True


class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Normalized adjacency matrix [N, N]
        Returns:
            Output features [N, out_features]
        """
        support = torch.mm(x, self.weight)  # [N, out_features]
        output = torch.mm(adj, support)  # [N, out_features]
        
        if self.bias is not None:
            output += self.bias
        
        return output


class GCNModel(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNLayer(n_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(GCNLayer(hidden_dim, n_classes))
        else:
            self.layers.append(nn.Linear(n_features, n_classes))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.layers[-1](x, adj)
        
        return x


class GNNAgent(Agent):
    """Agent with Graph Convolutional Network"""
    
    def __init__(self, agent_id: int, n_features: int, n_classes: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 lr: float = 0.01, epochs: int = 50, dropout: float = 0.0,
                 aggregation_strategy: Optional[AggregationStrategy] = None):
        super().__init__(agent_id, n_features, n_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.model = GCNModel(n_features, n_classes, hidden_dim, num_layers, dropout)
        self.trained = False
        self.device = torch.device('cpu')
        self.aggregation_strategy = aggregation_strategy or AverageAggregation()
        self.local_graph = None
        self.full_graph = None  # Store full graph for test predictions
    
    def set_local_graph(self, node_indices: np.ndarray, adj_matrix: np.ndarray):
        """Set local subgraph for this agent"""
        self.local_graph = {
            'node_indices': node_indices,
            'adj_matrix': adj_matrix
        }
    
    def _normalize_adj(self, adj: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix for GCN"""
        # Add self-loops
        adj = adj + np.eye(len(adj))
        # Compute degree matrix
        degree = np.array(adj.sum(1))
        degree_sqrt_inv = np.power(degree, -0.5).flatten()
        degree_sqrt_inv[degree_sqrt_inv == np.inf] = 0.0
        degree_sqrt_inv = np.diag(degree_sqrt_inv)
        # Normalized adjacency: D^(-1/2) * A * D^(-1/2)
        adj_normalized = degree_sqrt_inv @ adj @ degree_sqrt_inv
        return adj_normalized.astype(np.float32)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train GCN model"""
        if self.failed:
            return
        
        if len(X) == 0 or len(np.unique(y)) < 2:
            return
        
        if self.local_graph is None:
            adj = np.eye(len(X))
            self.set_local_graph(np.arange(len(X)), adj)
        
        try:
            adj = self.local_graph['adj_matrix']
            # Make symmetric
            adj = adj + adj.T
            adj = (adj > 0).astype(np.float32)
            adj_normalized = self._normalize_adj(adj)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            adj_tensor = torch.FloatTensor(adj_normalized).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor, adj_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            self.trained = True
        except Exception as e:
            print(f"GCN training error for agent {self.agent_id}: {e}")
            self.trained = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on local test data"""
        if self.failed or not self.trained:
            return np.zeros(len(X), dtype=int)
        
        try:
            # Use test graph if available (includes both training and test nodes)
            if hasattr(self, 'test_graph') and self.test_graph is not None:
                # Extract subgraph for test nodes only
                test_local_indices = self.test_graph['test_node_local_indices']
                full_adj = self.test_graph['adj_matrix']
                
                # Get adjacency for test nodes (includes connections to training nodes)
                adj = full_adj[np.ix_(test_local_indices, test_local_indices)]
                adj = adj + adj.T
                adj = (adj > 0).astype(np.float32)
            elif self.local_graph is not None and self.local_graph['adj_matrix'].shape[0] == len(X):
                # Use local graph if size matches
                adj = self.local_graph['adj_matrix']
                adj = adj + adj.T
                adj = (adj > 0).astype(np.float32)
            else:
                # Fallback to identity (no graph structure)
                adj = np.eye(len(X))
            
            adj_normalized = self._normalize_adj(adj)
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                adj_tensor = torch.FloatTensor(adj_normalized).to(self.device)
                outputs = self.model(X_tensor, adj_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        except Exception as e:
            print(f"GNN prediction error for agent {self.agent_id}: {e}")
            return np.zeros(len(X), dtype=int)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if not self.trained:
            return {'state_dict': None, 'trained': False}
        
        return {
            'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'trained': self.trained,
            'agent_id': self.agent_id
        }
    
    def update_from_messages(self, messages: List[Dict[str, Any]]):
        """Update model using aggregation strategy"""
        if self.failed or not messages:
            return
        
        own_params = None
        if self.trained:
            own_params = {
                'state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                'trained': self.trained
            }
        
        aggregated = self.aggregation_strategy.aggregate(messages, own_params)
        
        if aggregated and aggregated.get('trained', False):
            if 'state_dict' in aggregated and aggregated['state_dict'] is not None:
                self.model.load_state_dict(aggregated['state_dict'])
                self.trained = True

