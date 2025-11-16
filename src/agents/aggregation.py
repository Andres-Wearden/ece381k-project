"""Message aggregation strategies for multi-agent systems"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


def _normalize_coefficient_shapes(coef_list: List[np.ndarray], 
                                   intercept_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Normalize coefficient and intercept arrays to the same shape.
    
    Handles cases where agents trained on different subsets of classes,
    resulting in different coefficient shapes.
    
    Args:
        coef_list: List of coefficient arrays (may have different shapes)
        intercept_list: List of intercept arrays (may have different shapes)
        
    Returns:
        Tuple of (normalized_coefs, normalized_intercepts) with consistent shapes
    """
    if not coef_list:
        return [], []
    
    # Find the maximum shape (handles cases where agents saw different numbers of classes)
    shapes = [coef.shape for coef in coef_list]
    max_shape = max(shapes, key=lambda s: (len(s), s[0] if len(s) > 0 else 0))
    
    normalized_coefs = []
    normalized_intercepts = []
    
    for coef, intercept in zip(coef_list, intercept_list):
        if coef.shape != max_shape:
            # Pad or reshape to match max_shape
            if len(max_shape) == 2:  # 2D: (n_classes, n_features)
                n_classes, n_features = max_shape
                if len(coef.shape) == 1:  # 1D: (n_features,) - binary case
                    # Convert to 2D: (1, n_features) then pad to (n_classes, n_features)
                    padded = np.zeros(max_shape, dtype=coef.dtype)
                    padded[0, :] = coef
                    coef = padded
                    # Intercept: (1,) -> (n_classes,)
                    if intercept.ndim == 0:
                        intercept = np.array([intercept])
                    padded_intercept = np.zeros(n_classes, dtype=intercept.dtype)
                    padded_intercept[0] = intercept
                    intercept = padded_intercept
                elif len(coef.shape) == 2:  # 2D but different n_classes
                    # Pad with zeros for missing classes
                    padded = np.zeros(max_shape, dtype=coef.dtype)
                    padded[:coef.shape[0], :] = coef
                    coef = padded
                    # Intercept: pad similarly
                    if intercept.ndim == 0:
                        intercept = np.array([intercept])
                    padded_intercept = np.zeros(n_classes, dtype=intercept.dtype)
                    padded_intercept[:intercept.shape[0]] = intercept
                    intercept = padded_intercept
            else:  # 1D case
                # All should be 1D, just ensure same length
                if coef.shape[0] < max_shape[0]:
                    padded = np.zeros(max_shape, dtype=coef.dtype)
                    padded[:coef.shape[0]] = coef
                    coef = padded
                    if intercept.ndim == 0:
                        intercept = np.array([intercept])
                    if intercept.shape[0] < max_shape[0]:
                        padded_intercept = np.zeros(max_shape[0], dtype=intercept.dtype)
                        padded_intercept[:intercept.shape[0]] = intercept
                        intercept = padded_intercept
        
        normalized_coefs.append(coef)
        normalized_intercepts.append(intercept)
    
    return normalized_coefs, normalized_intercepts


class AggregationStrategy(ABC):
    """Base class for message aggregation strategies"""
    
    @abstractmethod
    def aggregate(self, messages: List[Dict[str, Any]], 
                  own_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregate messages from neighbors
        
        Args:
            messages: List of message dictionaries containing parameters
            own_params: Agent's own parameters (optional)
            
        Returns:
            Aggregated parameters
        """
        pass


class AverageAggregation(AggregationStrategy):
    """Simple averaging aggregation (baseline)"""
    
    def aggregate(self, messages: List[Dict[str, Any]], 
                  own_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Average all parameters equally"""
        # Messages come in format: {'params': {...}, 'weight': ..., 'sender': ..., 'receiver': ...}
        valid_messages = [msg for msg in messages 
                         if msg.get('params', {}).get('trained', False)]
        
        if not valid_messages:
            return own_params if own_params else {}
        
        params_list = [msg['params'] for msg in valid_messages]
        
        if own_params and own_params.get('trained', False):
            params_list.append(own_params)
        
        if not params_list:
            return {}
        
        # Determine parameter type
        first_params = params_list[0]
        
        if 'coef' in first_params:
            # Logistic/Linear agent
            coef_list = [p['coef'] for p in params_list if p.get('coef') is not None]
            intercept_list = [p['intercept'] for p in params_list if p.get('intercept') is not None]
            
            if coef_list:
                # Normalize all coefficients to the same shape
                normalized_coefs, normalized_intercepts = _normalize_coefficient_shapes(coef_list, intercept_list)
                
                # Now stack and compute mean
                coef_array = np.stack(normalized_coefs)
                intercept_array = np.stack(normalized_intercepts)
                
                return {
                    'coef': np.mean(coef_array, axis=0),
                    'intercept': np.mean(intercept_array, axis=0),
                    'trained': True
                }
        
        elif 'state_dict' in first_params:
            # Neural agent
            state_dicts = [p['state_dict'] for p in params_list if p.get('state_dict') is not None]
            if state_dicts:
                averaged_state = {}
                for key in state_dicts[0].keys():
                    averaged_state[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
                return {
                    'state_dict': averaged_state,
                    'trained': True
                }
        
        return {}


class WeightedAverageAggregation(AggregationStrategy):
    """Weighted averaging based on edge weights"""
    
    def aggregate(self, messages: List[Dict[str, Any]], 
                  own_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Weighted average using edge weights"""
        valid_messages = [msg for msg in messages 
                         if msg.get('params', {}).get('trained', False)]
        
        if not valid_messages:
            return own_params if own_params else {}
        
        # Extract weights (default to 1.0 if not present)
        weights = [msg.get('weight', 1.0) for msg in valid_messages]
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        params_list = [msg['params'] for msg in valid_messages]
        
        # Add own params with equal weight
        if own_params and own_params.get('trained', False):
            params_list.append(own_params)
            # Re-normalize weights
            weights = np.append(weights, 1.0 / len(valid_messages))
            weights = weights / weights.sum()
        
        if not params_list:
            return {}
        
        first_params = params_list[0]
        
        if 'coef' in first_params:
            coef_list = [p['coef'] for p in params_list if p.get('coef') is not None]
            intercept_list = [p['intercept'] for p in params_list if p.get('intercept') is not None]
            
            if coef_list:
                # Normalize all coefficients to the same shape
                normalized_coefs, normalized_intercepts = _normalize_coefficient_shapes(coef_list, intercept_list)
                
                # Stack and compute weighted average
                coef_array = np.stack(normalized_coefs)
                intercept_array = np.stack(normalized_intercepts)
                weighted_coef = np.average(coef_array, axis=0, weights=weights[:len(coef_list)])
                weighted_intercept = np.average(intercept_array, axis=0, weights=weights[:len(intercept_list)])
                return {
                    'coef': weighted_coef,
                    'intercept': weighted_intercept,
                    'trained': True
                }
        
        elif 'state_dict' in first_params:
            state_dicts = [p['state_dict'] for p in params_list if p.get('state_dict') is not None]
            if state_dicts:
                averaged_state = {}
                for key in state_dicts[0].keys():
                    stacked = torch.stack([sd[key].float() for sd in state_dicts])
                    # Weighted average
                    weights_tensor = torch.tensor(weights[:len(state_dicts)], dtype=torch.float32)
                    weights_tensor = weights_tensor.view(-1, *([1] * (stacked.dim() - 1)))
                    averaged_state[key] = (stacked * weights_tensor).sum(dim=0) / weights_tensor.sum()
                return {
                    'state_dict': averaged_state,
                    'trained': True
                }
        
        return {}


class AttentionAggregation(AggregationStrategy):
    """Attention-based aggregation using learned attention weights"""
    
    def __init__(self, n_features: int, hidden_dim: int = 32, temperature: float = 1.0):
        """
        Initialize attention mechanism
        
        Args:
            n_features: Number of input features
            hidden_dim: Hidden dimension for attention network
            temperature: Temperature for softmax (higher = softer)
        """
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.device = torch.device('cpu')
        
        # Simple attention network: computes attention scores
        # Input: concatenated parameters, Output: attention weight
        self.attention_net = nn.Sequential(
            nn.Linear(n_features * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(self.device)
        
    def _compute_attention_scores(self, params_list: List[Dict[str, Any]], 
                                   own_params: Optional[Dict[str, Any]]) -> np.ndarray:
        """Compute attention scores for each message"""
        if not params_list:
            return np.array([])
        
        # Extract parameter vectors for attention computation
        param_vectors = []
        
        for params in params_list:
            if 'coef' in params and params.get('coef') is not None:
                # Flatten coefficients
                coef = params['coef']
                if coef.ndim > 1:
                    coef = coef.flatten()
                param_vectors.append(coef)
            elif 'state_dict' in params and params.get('state_dict') is not None:
                # Extract first layer weights as representation
                state_dict = params['state_dict']
                if 'fc1.weight' in state_dict:
                    weights = state_dict['fc1.weight'].flatten().cpu().numpy()
                    param_vectors.append(weights)
                else:
                    # Fallback: use average of all parameters
                    all_params = torch.cat([v.flatten() for v in state_dict.values()]).cpu().numpy()
                    param_vectors.append(all_params)
            else:
                continue
        
        if not param_vectors:
            return np.ones(len(params_list)) / len(params_list)
        
        # Pad or truncate to same length
        max_len = max(len(v) for v in param_vectors)
        param_vectors_padded = []
        for v in param_vectors:
            if len(v) < max_len:
                v = np.pad(v, (0, max_len - len(v)), mode='constant')
            else:
                v = v[:max_len]
            param_vectors_padded.append(v)
        
        param_vectors = np.array(param_vectors_padded)
        
        # Use own params as query if available
        if own_params:
            if 'coef' in own_params and own_params.get('coef') is not None:
                query = own_params['coef']
                if query.ndim > 1:
                    query = query.flatten()
            elif 'state_dict' in own_params and own_params.get('state_dict') is not None:
                state_dict = own_params['state_dict']
                if 'fc1.weight' in state_dict:
                    query = state_dict['fc1.weight'].flatten().cpu().numpy()
                else:
                    all_params = torch.cat([v.flatten() for v in state_dict['state_dict'].values()]).cpu().numpy()
                    query = all_params
            else:
                query = np.mean(param_vectors, axis=0)
        else:
            query = np.mean(param_vectors, axis=0)
        
        # Pad query to match
        if len(query) < max_len:
            query = np.pad(query, (0, max_len - len(query)), mode='constant')
        else:
            query = query[:max_len]
        
        # Compute attention: concatenate query with each key
        attention_scores = []
        query_tensor = torch.FloatTensor(query).to(self.device)
        
        for param_vec in param_vectors:
            key_tensor = torch.FloatTensor(param_vec).to(self.device)
            # Concatenate query and key
            concat = torch.cat([query_tensor, key_tensor])
            # Ensure correct size
            if concat.size(0) > self.n_features * 2:
                concat = concat[:self.n_features * 2]
            elif concat.size(0) < self.n_features * 2:
                concat = torch.cat([concat, torch.zeros(self.n_features * 2 - concat.size(0)).to(self.device)])
            
            with torch.no_grad():
                score = self.attention_net(concat)
                attention_scores.append(score.item())
        
        # Apply softmax with temperature
        attention_scores = np.array(attention_scores)
        attention_scores = attention_scores / self.temperature
        attention_weights = np.exp(attention_scores - np.max(attention_scores))
        attention_weights = attention_weights / attention_weights.sum()
        
        return attention_weights
    
    def aggregate(self, messages: List[Dict[str, Any]], 
                  own_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Aggregate using attention mechanism"""
        valid_messages = [msg for msg in messages 
                         if msg.get('params', {}).get('trained', False)]
        
        if not valid_messages:
            return own_params if own_params else {}
        
        params_list = [msg['params'] for msg in valid_messages]
        
        # Compute attention weights
        attention_weights = self._compute_attention_scores(params_list, own_params)
        
        if len(attention_weights) == 0:
            return own_params if own_params else {}
        
        # Include own params with equal attention
        if own_params and own_params.get('trained', False):
            params_list.append(own_params)
            # Re-normalize to include own params
            attention_weights = np.append(attention_weights, np.mean(attention_weights))
            attention_weights = attention_weights / attention_weights.sum()
        
        if not params_list:
            return {}
        
        first_params = params_list[0]
        
        if 'coef' in first_params:
            coef_list = [p['coef'] for p in params_list if p.get('coef') is not None]
            intercept_list = [p['intercept'] for p in params_list if p.get('intercept') is not None]
            
            if coef_list:
                # Normalize all coefficients to the same shape
                normalized_coefs, normalized_intercepts = _normalize_coefficient_shapes(coef_list, intercept_list)
                
                # Stack and compute weighted average
                coef_array = np.stack(normalized_coefs)
                intercept_array = np.stack(normalized_intercepts)
                weighted_coef = np.average(coef_array, axis=0, weights=attention_weights[:len(coef_list)])
                weighted_intercept = np.average(intercept_array, axis=0, weights=attention_weights[:len(intercept_list)])
                return {
                    'coef': weighted_coef,
                    'intercept': weighted_intercept,
                    'trained': True
                }
        
        elif 'state_dict' in first_params:
            state_dicts = [p['state_dict'] for p in params_list if p.get('state_dict') is not None]
            if state_dicts:
                averaged_state = {}
                for key in state_dicts[0].keys():
                    stacked = torch.stack([sd[key].float() for sd in state_dicts])
                    weights_tensor = torch.tensor(attention_weights[:len(state_dicts)], dtype=torch.float32)
                    weights_tensor = weights_tensor.view(-1, *([1] * (stacked.dim() - 1)))
                    averaged_state[key] = (stacked * weights_tensor).sum(dim=0) / weights_tensor.sum()
                return {
                    'state_dict': averaged_state,
                    'trained': True
                }
        
        return {}


class PerformanceBasedAggregation(AggregationStrategy):
    """Aggregation weighted by sender's local performance"""
    
    def __init__(self, test_data: np.ndarray, test_labels: np.ndarray):
        """
        Initialize performance-based aggregation
        
        Args:
            test_data: Test data for evaluating sender performance
            test_labels: Test labels
        """
        self.test_data = test_data
        self.test_labels = test_labels
    
    def _evaluate_params(self, params: Dict[str, Any], agent) -> float:
        """Evaluate parameters on test data"""
        # This would require temporarily loading params into agent
        # For simplicity, we'll use a proxy: parameter magnitude/variance
        if 'coef' in params and params.get('coef') is not None:
            # Use coefficient magnitude as proxy for quality
            return np.linalg.norm(params['coef'])
        elif 'state_dict' in params and params.get('state_dict') is not None:
            # Use average parameter magnitude
            total_norm = sum(torch.norm(v).item() for v in params['state_dict'].values())
            return total_norm / len(params['state_dict'])
        return 0.0
    
    def aggregate(self, messages: List[Dict[str, Any]], 
                  own_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Weight by sender performance"""
        valid_messages = [msg for msg in messages 
                         if msg.get('params', {}).get('trained', False)]
        
        if not valid_messages:
            return own_params if own_params else {}
        
        params_list = [msg['params'] for msg in valid_messages]
        
        # Compute performance scores
        performance_scores = [self._evaluate_params(p, None) for p in params_list]
        performance_scores = np.array(performance_scores)
        
        # Normalize to weights
        if performance_scores.sum() > 0:
            weights = performance_scores / performance_scores.sum()
        else:
            weights = np.ones(len(performance_scores)) / len(performance_scores)
        
        if own_params and own_params.get('trained', False):
            params_list.append(own_params)
            own_score = self._evaluate_params(own_params, None)
            weights = np.append(weights, own_score)
            weights = weights / weights.sum()
        
        if not params_list:
            return {}
        
        first_params = params_list[0]
        
        if 'coef' in first_params:
            coef_list = [p['coef'] for p in params_list if p.get('coef') is not None]
            intercept_list = [p['intercept'] for p in params_list if p.get('intercept') is not None]
            
            if coef_list:
                # Normalize all coefficients to the same shape
                normalized_coefs, normalized_intercepts = _normalize_coefficient_shapes(coef_list, intercept_list)
                
                # Stack and compute weighted average
                coef_array = np.stack(normalized_coefs)
                intercept_array = np.stack(normalized_intercepts)
                weighted_coef = np.average(coef_array, axis=0, weights=weights[:len(coef_list)])
                weighted_intercept = np.average(intercept_array, axis=0, weights=weights[:len(intercept_list)])
                return {
                    'coef': weighted_coef,
                    'intercept': weighted_intercept,
                    'trained': True
                }
        
        elif 'state_dict' in first_params:
            state_dicts = [p['state_dict'] for p in params_list if p.get('state_dict') is not None]
            if state_dicts:
                averaged_state = {}
                for key in state_dicts[0].keys():
                    stacked = torch.stack([sd[key].float() for sd in state_dicts])
                    weights_tensor = torch.tensor(weights[:len(state_dicts)], dtype=torch.float32)
                    weights_tensor = weights_tensor.view(-1, *([1] * (stacked.dim() - 1)))
                    averaged_state[key] = (stacked * weights_tensor).sum(dim=0) / weights_tensor.sum()
                return {
                    'state_dict': averaged_state,
                    'trained': True
                }
        
        return {}


class MultiHeadAttentionAggregation(AggregationStrategy):
    """Multi-head attention-based aggregation for more sophisticated information exchange"""
    
    def __init__(self, n_features: int, n_heads: int = 4, hidden_dim: int = 64, temperature: float = 1.0):
        """
        Initialize multi-head attention mechanism
        
        Args:
            n_features: Number of input features
            n_heads: Number of attention heads
            hidden_dim: Hidden dimension for attention network
            temperature: Temperature for softmax
        """
        self.n_features = n_features
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.device = torch.device('cpu')
        
        # Multi-head attention: each head learns different aspects
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_heads)
        ]).to(self.device)
        
        # Combine heads
        self.head_combiner = nn.Linear(n_heads, 1).to(self.device)
    
    def _compute_multi_head_attention(self, params_list: List[Dict[str, Any]], 
                                      own_params: Optional[Dict[str, Any]]) -> np.ndarray:
        """Compute multi-head attention scores"""
        if not params_list:
            return np.array([])
        
        # Extract parameter vectors (same as single-head attention)
        param_vectors = []
        for params in params_list:
            if 'coef' in params and params.get('coef') is not None:
                coef = params['coef']
                if coef.ndim > 1:
                    coef = coef.flatten()
                param_vectors.append(coef)
            elif 'state_dict' in params and params.get('state_dict') is not None:
                state_dict = params['state_dict']
                if 'fc1.weight' in state_dict:
                    weights = state_dict['fc1.weight'].flatten().cpu().numpy()
                    param_vectors.append(weights)
                else:
                    all_params = torch.cat([v.flatten() for v in state_dict.values()]).cpu().numpy()
                    param_vectors.append(all_params)
            else:
                continue
        
        if not param_vectors:
            return np.ones(len(params_list)) / len(params_list)
        
        # Pad to same length
        max_len = max(len(v) for v in param_vectors)
        param_vectors_padded = []
        for v in param_vectors:
            if len(v) < max_len:
                v = np.pad(v, (0, max_len - len(v)), mode='constant')
            else:
                v = v[:max_len]
            param_vectors_padded.append(v)
        
        param_vectors = np.array(param_vectors_padded)
        
        # Get query (own params or mean)
        if own_params:
            if 'coef' in own_params and own_params.get('coef') is not None:
                query = own_params['coef']
                if query.ndim > 1:
                    query = query.flatten()
            elif 'state_dict' in own_params and own_params.get('state_dict') is not None:
                state_dict = own_params['state_dict']
                if 'fc1.weight' in state_dict:
                    query = state_dict['fc1.weight'].flatten().cpu().numpy()
                else:
                    all_params = torch.cat([v.flatten() for v in state_dict.values()]).cpu().numpy()
                    query = all_params
            else:
                query = np.mean(param_vectors, axis=0)
        else:
            query = np.mean(param_vectors, axis=0)
        
        if len(query) < max_len:
            query = np.pad(query, (0, max_len - len(query)), mode='constant')
        else:
            query = query[:max_len]
        
        query_tensor = torch.FloatTensor(query).to(self.device)
        
        # Compute attention for each head
        all_head_scores = []
        for head in self.attention_heads:
            head_scores = []
            for param_vec in param_vectors:
                key_tensor = torch.FloatTensor(param_vec).to(self.device)
                concat = torch.cat([query_tensor, key_tensor])
                
                if concat.size(0) > self.n_features * 2:
                    concat = concat[:self.n_features * 2]
                elif concat.size(0) < self.n_features * 2:
                    concat = torch.cat([concat, torch.zeros(self.n_features * 2 - concat.size(0)).to(self.device)])
                
                with torch.no_grad():
                    score = head(concat)
                    head_scores.append(score.item())
            all_head_scores.append(head_scores)
        
        # Combine heads
        all_head_scores = torch.FloatTensor(all_head_scores).T.to(self.device)  # [n_messages, n_heads]
        with torch.no_grad():
            combined_scores = self.head_combiner(all_head_scores).squeeze(-1).cpu().numpy()
        
        # Apply softmax with temperature
        combined_scores = combined_scores / self.temperature
        attention_weights = np.exp(combined_scores - np.max(combined_scores))
        attention_weights = attention_weights / attention_weights.sum()
        
        return attention_weights
    
    def aggregate(self, messages: List[Dict[str, Any]], 
                  own_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Aggregate using multi-head attention"""
        valid_messages = [msg for msg in messages 
                         if msg.get('params', {}).get('trained', False)]
        
        if not valid_messages:
            return own_params if own_params else {}
        
        params_list = [msg['params'] for msg in valid_messages]
        
        # Compute multi-head attention weights
        attention_weights = self._compute_multi_head_attention(params_list, own_params)
        
        if len(attention_weights) == 0:
            return own_params if own_params else {}
        
        # Include own params
        if own_params and own_params.get('trained', False):
            params_list.append(own_params)
            attention_weights = np.append(attention_weights, np.mean(attention_weights))
            attention_weights = attention_weights / attention_weights.sum()
        
        if not params_list:
            return {}
        
        first_params = params_list[0]
        
        if 'coef' in first_params:
            coef_list = [p['coef'] for p in params_list if p.get('coef') is not None]
            intercept_list = [p['intercept'] for p in params_list if p.get('intercept') is not None]
            
            if coef_list:
                # Normalize all coefficients to the same shape
                normalized_coefs, normalized_intercepts = _normalize_coefficient_shapes(coef_list, intercept_list)
                
                # Stack and compute weighted average
                coef_array = np.stack(normalized_coefs)
                intercept_array = np.stack(normalized_intercepts)
                weighted_coef = np.average(coef_array, axis=0, weights=attention_weights[:len(coef_list)])
                weighted_intercept = np.average(intercept_array, axis=0, weights=attention_weights[:len(intercept_list)])
                return {
                    'coef': weighted_coef,
                    'intercept': weighted_intercept,
                    'trained': True
                }
        
        elif 'state_dict' in first_params:
            state_dicts = [p['state_dict'] for p in params_list if p.get('state_dict') is not None]
            if state_dicts:
                averaged_state = {}
                for key in state_dicts[0].keys():
                    stacked = torch.stack([sd[key].float() for sd in state_dicts])
                    weights_tensor = torch.tensor(attention_weights[:len(state_dicts)], dtype=torch.float32)
                    weights_tensor = weights_tensor.view(-1, *([1] * (stacked.dim() - 1)))
                    averaged_state[key] = (stacked * weights_tensor).sum(dim=0) / weights_tensor.sum()
                return {
                    'state_dict': averaged_state,
                    'trained': True
                }
        
        return {}


def get_aggregation_strategy(strategy_name: str, **kwargs) -> AggregationStrategy:
    """
    Factory function to get aggregation strategy
    
    Args:
        strategy_name: Name of strategy ('average', 'weighted', 'attention', 'performance')
        **kwargs: Strategy-specific parameters
        
    Returns:
        AggregationStrategy instance
    """
    strategies = {
        'average': AverageAggregation,
        'weighted': WeightedAverageAggregation,
        'attention': AttentionAggregation,
        'multi_head_attention': MultiHeadAttentionAggregation,
        'performance': PerformanceBasedAggregation
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown aggregation strategy: {strategy_name}. "
                        f"Available: {list(strategies.keys())}")
    
    strategy_class = strategies[strategy_name]
    return strategy_class(**kwargs)

