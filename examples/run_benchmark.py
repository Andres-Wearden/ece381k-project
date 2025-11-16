#!/usr/bin/env python3
"""
Example script for running multi-agent benchmarks programmatically
"""

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import load_dataset
from src.agents import LogisticAgent, NeuralAgent
from src.networks import build_topology
from src.simulation import SimulationEngine, NodeFailure
from src.evaluation import evaluate_system
from src.utils import plot_accuracy_history, plot_network, generate_report


def run_simple_benchmark():
    """Run a simple benchmark example"""
    
    print("="*60)
    print("Simple Multi-Agent Benchmark Example")
    print("="*60)
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    graph, features, labels = load_dataset('synthetic_classification', 
                                           n_nodes=80, n_features=10, n_classes=2)
    
    # Split data
    train_idx, test_idx = train_test_split(range(len(features)), test_size=0.3, 
                                           random_state=42, stratify=labels)
    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]
    
    print(f"   Dataset: {len(features)} samples, {features.shape[1]} features")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 2. Create network topology
    print("\n2. Building star topology...")
    n_agents = 10
    network = build_topology('star', n_agents, hub_id=0, bidirectional=True)
    print(f"   Network: {n_agents} agents, {network.number_of_edges()} edges")
    
    # 3. Create agents
    print("\n3. Creating agents...")
    agents = [LogisticAgent(i, features.shape[1], len(np.unique(labels))) 
              for i in range(n_agents)]
    
    # 4. Setup simulation
    print("\n4. Setting up simulation...")
    sim = SimulationEngine(agents, network, X_test, y_test)
    sim.distribute_data(X_train, y_train, distribution='equal')
    
    # Add perturbation
    sim.add_perturbation(NodeFailure(failure_prob=0.1, failure_duration=5, start_time=10))
    print("   Added node failure perturbation")
    
    # 5. Run simulation
    print("\n5. Running simulation (15 rounds)...")
    results = sim.run(n_rounds=15, verbose=True)
    
    # 6. Evaluate
    print("\n6. Evaluating system...")
    eval_results = evaluate_system(agents, network, X_test, y_test,
                                   results['history']['accuracy'],
                                   results['history']['failed_nodes'])
    
    # 7. Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"Average Accuracy: {results['average_accuracy']:.4f}")
    print(f"Robustness: {eval_results['robustness']:.4f}")
    print(f"Max Failed Nodes: {results['max_failed_nodes']}")
    print(f"Total Messages: {results['total_messages']}")
    
    # 8. Save visualizations
    print("\n7. Saving visualizations...")
    os.makedirs('outputs', exist_ok=True)
    plot_accuracy_history(results['history'], save_path='outputs/example_accuracy.png')
    plot_network(network, agents, save_path='outputs/example_network.png', 
                title="Star Topology Example")
    print("   Saved to outputs/")
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)
    
    return results


def compare_topologies():
    """Compare different topologies"""
    
    print("\n" + "="*60)
    print("Comparing Multiple Topologies")
    print("="*60)
    
    # Load common dataset
    graph, features, labels = load_dataset('synthetic_moons', n_nodes=60, n_features=5)
    train_idx, test_idx = train_test_split(range(len(features)), test_size=0.3, 
                                           random_state=42, stratify=labels)
    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]
    
    topologies = ['star', 'cascade', 'feedback_rewired']
    all_results = {}
    
    for topo in topologies:
        print(f"\nRunning {topo} topology...")
        
        # Build topology
        network = build_topology(topo, n_agents=10)
        
        # Create agents
        agents = [LogisticAgent(i, features.shape[1], len(np.unique(labels))) 
                 for i in range(10)]
        
        # Simulate
        sim = SimulationEngine(agents, network, X_test, y_test)
        sim.distribute_data(X_train, y_train)
        results = sim.run(n_rounds=15, verbose=False)
        
        # Evaluate
        eval_results = evaluate_system(agents, network, X_test, y_test,
                                       results['history']['accuracy'],
                                       results['history']['failed_nodes'])
        results.update(eval_results)
        
        all_results[topo] = results
        print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    config = {'dataset': 'synthetic_moons', 'agent_type': 'logistic', 
              'n_agents': 10, 'n_rounds': 15}
    generate_report(all_results, config, 'outputs/comparison_report.md')
    
    print("\n" + "="*60)
    print("Comparison complete! Check outputs/comparison_report.md")
    print("="*60)


if __name__ == '__main__':
    # Run simple benchmark
    run_simple_benchmark()
    
    # Uncomment to compare topologies
    # compare_topologies()

