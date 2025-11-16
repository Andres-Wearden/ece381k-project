#!/usr/bin/env python3
"""Command-line interface for multi-agent benchmarking"""

import argparse
import yaml
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.datasets import load_dataset
from src.agents import (LogisticAgent, LinearAgent, NeuralAgent, GATAgent, GNNAgent,
                       GraphSAGEAgent, GINAgent, RandomForestAgent)
from src.agents.aggregation import get_aggregation_strategy
from src.networks import build_topology
from src.simulation import SimulationEngine, NodeFailure, DelayPerturbation
from src.evaluation import evaluate_system
from src.utils import (generate_report, save_results_to_csv, 
                      plot_accuracy_history, plot_network, 
                      plot_comparison, create_summary_plots,
                      create_comprehensive_comparison)


AGENT_TYPES = {
    'logistic': LogisticAgent,
    'linear': LinearAgent,
    'neural': NeuralAgent,
    'gat': GATAgent,
    'gnn': GNNAgent,
    'graphsage': GraphSAGEAgent,
    'gin': GINAgent,
    'rf': RandomForestAgent,
    'random_forest': RandomForestAgent
}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_experiment(config: dict, verbose: bool = True):
    """Run a single experiment based on configuration"""
    
    # Load dataset
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment: {config['name']}")
        print(f"{'='*60}")
        print(f"Loading dataset: {config['dataset']}...")
    
    dataset_params = config.get('dataset_params', {})
    graph, features, labels = load_dataset(config['dataset'], **dataset_params)
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(features)), 
        test_size=config.get('test_size', 0.3),
        random_state=42,
        stratify=labels
    )
    
    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]
    
    n_features = features.shape[1]
    n_classes = len(np.unique(labels))
    n_agents = config.get('n_agents', len(graph.nodes()))
    
    if verbose:
        print(f"Dataset: {len(features)} samples, {n_features} features, {n_classes} classes")
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build topology
    if verbose:
        print(f"Building {config['topology']} topology with {n_agents} agents...")
    
    topology_params = config.get('topology_params', {})
    network = build_topology(config['topology'], n_agents, **topology_params)
    
    # Create agents
    if verbose:
        print(f"Creating {config['agent_type']} agents...")
    
    agent_class = AGENT_TYPES[config['agent_type']]
    agent_params = config.get('agent_params', {}).copy()
    
    # Handle aggregation strategy
    aggregation_config = config.get('aggregation', {})
    if aggregation_config:
        agg_strategy_name = aggregation_config.get('strategy', 'average')
        agg_strategy_params = aggregation_config.get('params', {})
        
        # Add n_features for attention aggregation
        if agg_strategy_name in ['attention', 'multi_head_attention']:
            agg_strategy_params['n_features'] = n_features
        
        # Add test data for performance-based aggregation
        if agg_strategy_name == 'performance':
            # Will be set after test data is available
            pass
        
        aggregation_strategy = get_aggregation_strategy(agg_strategy_name, **agg_strategy_params)
        agent_params['aggregation_strategy'] = aggregation_strategy
        
        if verbose:
            print(f"Using aggregation strategy: {agg_strategy_name}")
    
    agents = [
        agent_class(i, n_features, n_classes, **agent_params)
        for i in range(n_agents)
    ]
    
    # Update performance-based aggregation with test data if needed
    if aggregation_config and aggregation_config.get('strategy') == 'performance':
        for agent in agents:
            if hasattr(agent, 'aggregation_strategy') and hasattr(agent.aggregation_strategy, 'test_data'):
                agent.aggregation_strategy.test_data = X_test
                agent.aggregation_strategy.test_labels = y_test
    
    # Create simulation (pass graph structure for GAT/GNN agents)
    data_graph = graph if config['agent_type'] in ['gat', 'gnn', 'graphsage', 'gin'] else None
    sim = SimulationEngine(agents, network, X_test, y_test, data_graph=data_graph, 
                          test_indices=np.array(test_idx))
    
    # Distribute training data
    distribution = config.get('data_distribution', 'equal')
    samples_per_agent = config.get('samples_per_agent', None)
    sim.distribute_data(X_train, y_train, distribution=distribution, 
                       samples_per_agent=samples_per_agent,
                       train_indices=np.array(train_idx))
    
    # Add perturbations
    perturbations = config.get('perturbations', [])
    for pert_config in perturbations:
        if pert_config['type'] == 'node_failure':
            pert = NodeFailure(**pert_config.get('params', {}))
            sim.add_perturbation(pert)
            if verbose:
                print(f"Added perturbation: {pert.get_description()}")
        elif pert_config['type'] == 'delay':
            pert = DelayPerturbation(**pert_config.get('params', {}))
            sim.add_perturbation(pert)
            if verbose:
                print(f"Added perturbation: {pert.get_description()}")
    
    # Run simulation
    n_rounds = config.get('n_rounds', 20)
    if verbose:
        print(f"\nRunning simulation for {n_rounds} rounds...")
    
    results = sim.run(n_rounds, verbose=verbose)
    
    # Detailed evaluation
    eval_results = evaluate_system(
        agents, network, X_test, y_test,
        results['history']['accuracy'],
        results['history']['failed_nodes']
    )
    
    # Combine results
    results.update(eval_results)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
        print(f"  Average Accuracy: {results['average_accuracy']:.4f}")
        print(f"  Robustness: {results['robustness']:.4f}")
    
    return results, sim


def run_benchmark(config_path: str, output_dir: str = 'outputs', verbose: bool = True):
    """Run full benchmark from config file"""
    
    # Load config
    config = load_config(config_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiments
    all_results = {}
    all_sims = {}
    
    experiments = config.get('experiments', [config])
    
    for exp_config in experiments:
        exp_name = exp_config['name']
        results, sim = run_single_experiment(exp_config, verbose=verbose)
        all_results[exp_name] = results
        all_sims[exp_name] = sim
        
        # Save individual results
        csv_path = os.path.join(output_dir, f'{exp_name}_results.csv')
        save_results_to_csv(results, csv_path)
        
        # Plot accuracy history
        plot_path = os.path.join(output_dir, f'{exp_name}_accuracy.png')
        plot_accuracy_history(results['history'], save_path=plot_path)
        
        # Plot network
        network_path = os.path.join(output_dir, f'{exp_name}_network.png')
        plot_network(sim.network, sim.agents, save_path=network_path, 
                    title=f"{exp_name} Network Topology")
    
    # Generate comparison plots if multiple experiments
    if len(all_results) > 1:
        comparison_path = os.path.join(output_dir, 'comparison.png')
        plot_comparison(all_results, save_path=comparison_path)
        
        create_summary_plots(all_results, output_dir)
        
        # Create comprehensive comparison if we have many experiments (likely all models × all topologies)
        if len(all_results) >= 20:  # Threshold for comprehensive comparison
            create_comprehensive_comparison(all_results, output_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, 'benchmark_report.md')
    generate_report(all_results, config, report_path)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Agent AI Systems Benchmarking Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run star topology benchmark
  python cli.py --config configs/star_config.yaml
  
  # Run cascade topology with custom output directory
  python cli.py --config configs/cascade_config.yaml --output results/cascade
  
  # Run all topologies comparison
  python cli.py --config configs/all_topologies.yaml
  
  # Quick test with synthetic data
  python cli.py --quick-test
        """
    )
    
    parser.add_argument('--config', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with default configuration')
    
    args = parser.parse_args()
    
    if args.quick_test:
        # Create quick test config
        quick_config = {
            'name': 'quick_test',
            'dataset': 'synthetic_classification',
            'dataset_params': {'n_nodes': 50, 'n_features': 10},
            'topology': 'star',
            'n_agents': 10,
            'agent_type': 'logistic',
            'n_rounds': 10,
            'test_size': 0.3
        }
        print("Running quick test...")
        results, sim = run_single_experiment(quick_config, verbose=not args.quiet)
        
        # Save quick results
        os.makedirs(args.output, exist_ok=True)
        save_results_to_csv(results, os.path.join(args.output, 'quick_test.csv'))
        plot_accuracy_history(results['history'], 
                            save_path=os.path.join(args.output, 'quick_test_accuracy.png'))
        print(f"\nQuick test complete! Results saved to {args.output}")
        
    elif args.config:
        run_benchmark(args.config, args.output, verbose=not args.quiet)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

