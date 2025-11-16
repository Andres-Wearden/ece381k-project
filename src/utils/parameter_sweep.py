"""Parameter sweep utilities for hyperparameter optimization"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional
from itertools import product
import json
import os
from tqdm import tqdm


class ParameterSweep:
    """Run parameter sweeps to find optimal hyperparameters"""
    
    def __init__(self, param_grid: Dict[str, List[Any]], 
                 objective_function: Callable,
                 maximize: bool = True,
                 n_trials: Optional[int] = None):
        """
        Initialize parameter sweep
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            objective_function: Function that takes params dict and returns score
            maximize: If True, maximize objective; if False, minimize
            n_trials: If specified, randomly sample n_trials combinations instead of full grid
        """
        self.param_grid = param_grid
        self.objective_function = objective_function
        self.maximize = maximize
        self.n_trials = n_trials
        self.results = []
        self.best_params = None
        self.best_score = None
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        if self.n_trials is not None:
            # Random sampling
            combinations = []
            param_names = list(self.param_grid.keys())
            param_values = list(self.param_grid.values())
            
            for _ in range(self.n_trials):
                combo = {}
                for name, values in zip(param_names, param_values):
                    combo[name] = np.random.choice(values)
                combinations.append(combo)
            return combinations
        else:
            # Full grid search
            param_names = list(self.param_grid.keys())
            param_values = list(self.param_grid.values())
            combinations = []
            for combo in product(*param_values):
                combinations.append(dict(zip(param_names, combo)))
            return combinations
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run parameter sweep
        
        Returns:
            Dictionary with best parameters and all results
        """
        combinations = self._generate_param_combinations()
        
        if verbose:
            print(f"Running parameter sweep: {len(combinations)} combinations")
            iterator = tqdm(combinations, desc="Parameter Sweep")
        else:
            iterator = combinations
        
        for params in iterator:
            try:
                score = self.objective_function(params)
                self.results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                # Update best
                if self.best_score is None:
                    self.best_score = score
                    self.best_params = params.copy()
                elif (self.maximize and score > self.best_score) or \
                     (not self.maximize and score < self.best_score):
                    self.best_score = score
                    self.best_params = params.copy()
                
                if verbose:
                    iterator.set_postfix({'best': f'{self.best_score:.4f}'})
            except Exception as e:
                if verbose:
                    print(f"Error with params {params}: {e}")
                continue
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'n_trials': len(self.results)
        }
    
    def get_top_k(self, k: int = 5) -> List[Dict[str, Any]]:
        """Get top k parameter combinations"""
        sorted_results = sorted(
            self.results,
            key=lambda x: x['score'],
            reverse=self.maximize
        )
        return sorted_results[:k]
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable({
            'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score is not None else None,
            'all_results': self.results,
            'n_trials': len(self.results)
        })
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filepath: str):
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.best_params = data['best_params']
        self.best_score = data['best_score']
        self.results = data['all_results']


def create_hyperparameter_grid(agent_type: str = 'logistic') -> Dict[str, List[Any]]:
    """
    Create default hyperparameter grid for different agent types
    
    Args:
        agent_type: Type of agent ('logistic', 'linear', 'neural')
        
    Returns:
        Parameter grid dictionary
    """
    if agent_type == 'logistic':
        return {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'max_iter': [500, 1000, 2000],
            'aggregation': ['average', 'weighted', 'attention'],
            'samples_per_agent': [50, 100, 200, 500],
            'n_rounds': [10, 20, 30]
        }
    elif agent_type == 'linear':
        return {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'aggregation': ['average', 'weighted', 'attention'],
            'samples_per_agent': [50, 100, 200, 500],
            'n_rounds': [10, 20, 30]
        }
    elif agent_type == 'neural':
        return {
            'hidden_dim': [16, 32, 64],
            'lr': [0.001, 0.01, 0.1],
            'epochs': [20, 50, 100],
            'aggregation': ['average', 'weighted', 'attention'],
            'samples_per_agent': [50, 100, 200, 500],
            'n_rounds': [10, 20, 30]
        }
    else:
        return {
            'samples_per_agent': [50, 100, 200, 500],
            'n_rounds': [10, 20, 30]
        }
