"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import Dict, Any, List
import os


sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_accuracy_history(history: Dict[str, List], save_path: str = None, title: str = None):
    """
    Plot accuracy and failure history over time
    
    Args:
        history: Dictionary with 'accuracy' and 'failed_nodes' keys
        save_path: Path to save figure
        title: Optional title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Accuracy plot
    rounds = range(len(history['accuracy']))
    ax1.plot(rounds, history['accuracy'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    plot_title = title if title else 'System Accuracy Over Time'
    ax1.set_title(plot_title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Failed nodes plot
    ax2.plot(rounds, history['failed_nodes'], linewidth=2, color='#A23B72')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Failed Nodes', fontsize=12)
    ax2.set_title('Failed Nodes Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_network(network: nx.DiGraph, agents: List = None, 
                 save_path: str = None, title: str = "Network Topology"):
    """
    Visualize network topology
    
    Args:
        network: NetworkX DiGraph
        agents: List of agents (to show failed nodes)
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Determine node colors
    if agents:
        node_colors = ['#FF6B6B' if agent.failed else '#4ECDC4' 
                      for agent in agents]
    else:
        node_colors = '#4ECDC4'
    
    # Layout
    if network.number_of_nodes() < 50:
        pos = nx.spring_layout(network, k=1, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(network, k=2, iterations=30, seed=42)
    
    # Draw network
    nx.draw_networkx_nodes(network, pos, node_color=node_colors, 
                          node_size=500, alpha=0.9, ax=ax)
    
    # Draw edges with varying widths based on weights
    edges = network.edges()
    weights = [network[u][v].get('weight', 1.0) for u, v in edges]
    nx.draw_networkx_edges(network, pos, edgelist=edges, 
                          width=[w*2 for w in weights],
                          alpha=0.5, edge_color='#95A5A6',
                          arrows=True, arrowsize=15, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(network, pos, font_size=8, 
                           font_weight='bold', ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    if agents:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ECDC4', label='Active'),
            Patch(facecolor='#FF6B6B', label='Failed')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(results: Dict[str, Any], save_path: str = None):
    """
    Plot comparison of different topologies/configurations
    
    Args:
        results: Dictionary mapping config names to result dictionaries
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = list(results.keys())
    
    # Accuracy comparison
    accuracies = [results[c]['final_accuracy'] for c in configs]
    axes[0, 0].bar(configs, accuracies, color='#2E86AB', alpha=0.8)
    axes[0, 0].set_ylabel('Final Accuracy', fontsize=11)
    axes[0, 0].set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Robustness comparison
    if 'robustness' in results[configs[0]]:
        robustness = [results[c].get('robustness', 0) for c in configs]
        axes[0, 1].bar(configs, robustness, color='#F18F01', alpha=0.8)
        axes[0, 1].set_ylabel('Robustness', fontsize=11)
        axes[0, 1].set_title('Robustness Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim([0, 1.05])
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Error depth comparison
    if 'error_depth' in results[configs[0]]:
        error_depths = [results[c]['error_depth']['mean_depth'] for c in configs]
        axes[1, 0].bar(configs, error_depths, color='#A23B72', alpha=0.8)
        axes[1, 0].set_ylabel('Mean Error Depth', fontsize=11)
        axes[1, 0].set_title('Error Propagation Depth', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Network efficiency
    if 'network_info' in results[configs[0]]:
        densities = [results[c]['network_info']['density'] for c in configs]
        axes[1, 1].bar(configs, densities, color='#06A77D', alpha=0.8)
        axes[1, 1].set_ylabel('Network Density', fontsize=11)
        axes[1, 1].set_title('Network Connectivity', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_summary_plots(all_results: Dict[str, Any], output_dir: str):
    """
    Create comprehensive summary plots
    
    Args:
        all_results: Dictionary with all experimental results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy histories for each topology
    fig, ax = plt.subplots(figsize=(12, 6))
    for config_name, results in all_results.items():
        if 'history' in results:
            history = results['history']
            rounds = range(len(history['accuracy']))
            ax.plot(rounds, history['accuracy'], linewidth=2, 
                   label=config_name, alpha=0.8)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Evolution Across Topologies', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Performance metrics heatmap
    metrics_names = ['final_accuracy', 'average_accuracy', 'robustness']
    config_names = list(all_results.keys())
    
    metrics_matrix = []
    available_metrics = []
    
    for metric in metrics_names:
        if metric in all_results[config_names[0]]:
            available_metrics.append(metric)
            row = [all_results[c].get(metric, 0) for c in config_names]
            metrics_matrix.append(row)
    
    if metrics_matrix:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', 
                   xticklabels=config_names, yticklabels=available_metrics,
                   cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_comprehensive_comparison(all_results: Dict[str, Any], output_dir: str):
    """
    Create comprehensive comparison plots for all models × all topologies
    
    Args:
        all_results: Dictionary with all experimental results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse experiment names to extract model and topology
    models = ['logistic', 'neural', 'gat', 'rf']
    topologies = ['star', 'cascade', 'feedback_rewired', 'ring', 'mesh', 
                  'small_world', 'scale_free', 'tree', 'grid']
    
    # Build matrices: model × topology
    accuracy_matrix = np.zeros((len(models), len(topologies)))
    robustness_matrix = np.zeros((len(models), len(topologies)))
    avg_accuracy_matrix = np.zeros((len(models), len(topologies)))
    
    # Fill matrices
    for exp_name, results in all_results.items():
        # Parse experiment name (format: topology_model)
        parts = exp_name.split('_')
        if len(parts) >= 2:
            # Find topology (first part or first few parts)
            topology = None
            model = None
            
            for topo in topologies:
                if exp_name.startswith(topo):
                    topology = topo
                    # Model is everything after topology
                    model_part = exp_name[len(topo)+1:] if len(exp_name) > len(topo) else exp_name
                    break
            
            for mod in models:
                if exp_name.endswith(mod) or mod in exp_name:
                    model = mod
                    break
            
            if topology and model:
                topo_idx = topologies.index(topology)
                model_idx = models.index(model)
                
                accuracy_matrix[model_idx, topo_idx] = results.get('final_accuracy', 0)
                robustness_matrix[model_idx, topo_idx] = results.get('robustness', 0)
                avg_accuracy_matrix[model_idx, topo_idx] = results.get('average_accuracy', 0)
    
    # ========== PLOT 1: Accuracy Heatmap (Model × Topology) ==========
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', 
               xticklabels=[t.replace('_', ' ').title() for t in topologies],
               yticklabels=[m.title() for m in models],
               cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Final Accuracy'},
               vmin=0, vmax=1)
    ax.set_title('Final Accuracy: Models × Topologies', fontsize=16, fontweight='bold')
    ax.set_xlabel('Network Topology', fontsize=12)
    ax.set_ylabel('Agent Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap_models_topologies.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== PLOT 2: Robustness Heatmap ==========
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(robustness_matrix, annot=True, fmt='.3f', 
               xticklabels=[t.replace('_', ' ').title() for t in topologies],
               yticklabels=[m.title() for m in models],
               cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Robustness'},
               vmin=0, vmax=1)
    ax.set_title('Robustness: Models × Topologies', fontsize=16, fontweight='bold')
    ax.set_xlabel('Network Topology', fontsize=12)
    ax.set_ylabel('Agent Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_heatmap_models_topologies.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== PLOT 3: Average Accuracy by Model ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    model_avg_acc = np.mean(avg_accuracy_matrix, axis=1)
    bars = ax.bar(range(len(models)), model_avg_acc, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.title() for m in models], rotation=45, ha='right')
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Average Accuracy Across All Topologies by Model', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, model_avg_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_accuracy_by_model.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== PLOT 4: Average Accuracy by Topology ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    topo_avg_acc = np.mean(avg_accuracy_matrix, axis=0)
    bars = ax.bar(range(len(topologies)), topo_avg_acc,
                  color=plt.cm.plasma(np.linspace(0, 1, len(topologies))))
    ax.set_xticks(range(len(topologies)))
    ax.set_xticklabels([t.replace('_', ' ').title() for t in topologies], 
                       rotation=45, ha='right')
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Average Accuracy Across All Models by Topology', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, topo_avg_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_accuracy_by_topology.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== PLOT 5: Accuracy Evolution by Model ==========
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        for topo_idx, topology in enumerate(topologies):
            exp_name = f"{topology}_{model}"
            if exp_name in all_results and 'history' in all_results[exp_name]:
                history = all_results[exp_name]['history']
                rounds = range(len(history['accuracy']))
                ax.plot(rounds, history['accuracy'], 
                       label=topology.replace('_', ' ').title(), 
                       linewidth=2, alpha=0.7)
        
        ax.set_title(f'{model.title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        if model_idx == 0:  # Only show legend on first plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Hide unused subplot
    axes[-1].axis('off')
    
    plt.suptitle('Accuracy Evolution by Model Across All Topologies', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_evolution_by_model.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== PLOT 6: Accuracy Evolution by Topology ==========
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for topo_idx, topology in enumerate(topologies):
        ax = axes[topo_idx]
        for model_idx, model in enumerate(models):
            exp_name = f"{topology}_{model}"
            if exp_name in all_results and 'history' in all_results[exp_name]:
                history = all_results[exp_name]['history']
                rounds = range(len(history['accuracy']))
                ax.plot(rounds, history['accuracy'], 
                       label=model.title(), 
                       linewidth=2, alpha=0.7)
        
        ax.set_title(topology.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        if topo_idx == 0:  # Only show legend on first plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Accuracy Evolution by Topology Across All Models', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_evolution_by_topology.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== PLOT 7: Best Combinations ==========
    # Find top 10 best combinations
    best_combinations = []
    for exp_name, results in all_results.items():
        parts = exp_name.split('_')
        topology = None
        model = None
        
        for topo in topologies:
            if exp_name.startswith(topo):
                topology = topo
                break
        
        for mod in models:
            if exp_name.endswith(mod) or mod in exp_name:
                model = mod
                break
        
        if topology and model:
            best_combinations.append({
                'name': exp_name,
                'topology': topology,
                'model': model,
                'accuracy': results.get('final_accuracy', 0),
                'robustness': results.get('robustness', 0)
            })
    
    # Sort by accuracy
    best_combinations.sort(key=lambda x: x['accuracy'], reverse=True)
    top_10 = best_combinations[:10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 10 by accuracy
    names = [f"{c['topology'].replace('_', ' ').title()}\n{c['model'].title()}" 
             for c in top_10]
    accuracies = [c['accuracy'] for c in top_10]
    bars = ax1.barh(range(len(names)), accuracies, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(names))))
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Final Accuracy', fontsize=12)
    ax1.set_title('Top 10 Model-Topology Combinations\n(by Accuracy)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    # Top 10 by robustness
    best_combinations.sort(key=lambda x: x['robustness'], reverse=True)
    top_10_rob = best_combinations[:10]
    names_rob = [f"{c['topology'].replace('_', ' ').title()}\n{c['model'].title()}" 
                 for c in top_10_rob]
    robustnesses = [c['robustness'] for c in top_10_rob]
    bars = ax2.barh(range(len(names_rob)), robustnesses,
                    color=plt.cm.plasma(np.linspace(0, 1, len(names_rob))))
    ax2.set_yticks(range(len(names_rob)))
    ax2.set_yticklabels(names_rob, fontsize=9)
    ax2.set_xlabel('Robustness', fontsize=12)
    ax2.set_title('Top 10 Model-Topology Combinations\n(by Robustness)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, robustnesses)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_combinations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created comprehensive comparison plots in {output_dir}")

