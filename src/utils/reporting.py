"""Reporting utilities for generating analysis reports"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
from datetime import datetime


def save_results_to_csv(results: Dict[str, Any], output_path: str):
    """
    Save results to CSV file
    
    Args:
        results: Results dictionary
        output_path: Path to save CSV
    """
    # Flatten results for CSV
    flat_results = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str)):
                    flat_results[f"{key}_{sub_key}"] = sub_value
                elif isinstance(sub_value, dict):
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        if isinstance(sub_sub_value, (int, float, str)):
                            flat_results[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_value
        elif isinstance(value, (int, float, str)):
            flat_results[key] = value
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], (int, float)):
                flat_results[f"{key}_mean"] = np.mean(value)
                flat_results[f"{key}_std"] = np.std(value)
                flat_results[f"{key}_final"] = value[-1]
    
    df = pd.DataFrame([flat_results])
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def generate_design_rules(all_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Generate design rules based on experimental results
    
    Args:
        all_results: Dictionary mapping configuration names to results
        
    Returns:
        List of design rule strings
    """
    rules = []
    
    # Extract metrics
    configs = list(all_results.keys())
    accuracies = {c: all_results[c].get('final_accuracy', 0) for c in configs}
    robustness = {c: all_results[c].get('robustness', 0) for c in configs}
    
    # Rule 1: Best topology for accuracy
    best_accuracy_config = max(accuracies, key=accuracies.get)
    rules.append(
        f"🏆 **Best for Accuracy**: {best_accuracy_config} achieved {accuracies[best_accuracy_config]:.3f} accuracy. "
        f"This topology is recommended when prediction quality is the primary concern."
    )
    
    # Rule 2: Best topology for robustness
    if robustness:
        best_robustness_config = max(robustness, key=robustness.get)
        rules.append(
            f"🛡️ **Best for Robustness**: {best_robustness_config} achieved {robustness[best_robustness_config]:.3f} robustness. "
            f"Use this topology in environments with frequent failures or perturbations."
        )
    
    # Rule 3: Network size impact
    network_sizes = {c: all_results[c]['network_info']['n_agents'] for c in configs}
    size_accuracy_corr = np.corrcoef(
        [network_sizes[c] for c in configs],
        [accuracies[c] for c in configs]
    )[0, 1]
    
    if size_accuracy_corr > 0.3:
        rules.append(
            f"📈 **Scale Positively**: Larger networks show improved accuracy (correlation: {size_accuracy_corr:.2f}). "
            f"Consider scaling up agent count for better performance."
        )
    elif size_accuracy_corr < -0.3:
        rules.append(
            f"📉 **Diminishing Returns**: Larger networks show reduced accuracy (correlation: {size_accuracy_corr:.2f}). "
            f"Avoid over-scaling as it may introduce coordination overhead."
        )
    else:
        rules.append(
            f"⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: {size_accuracy_corr:.2f}). "
            f"Focus on topology design rather than scaling."
        )
    
    # Rule 4: Failure impact
    max_failures = {c: all_results[c].get('max_failed_nodes', 0) for c in configs}
    if any(max_failures.values()):
        most_resilient = min(configs, key=lambda c: accuracies[c] / (max_failures[c] + 1))
        rules.append(
            f"💪 **Failure Resilience**: {most_resilient} maintains performance best under node failures. "
            f"This topology has effective redundancy and graceful degradation."
        )
    
    # Rule 5: Communication efficiency
    densities = {c: all_results[c]['network_info']['density'] for c in configs}
    avg_accuracy = np.mean(list(accuracies.values()))
    efficient_configs = [c for c in configs if accuracies[c] >= avg_accuracy and densities[c] < np.mean(list(densities.values()))]
    
    if efficient_configs:
        rules.append(
            f"⚡ **Communication Efficient**: {', '.join(efficient_configs)} achieve above-average accuracy "
            f"with below-average network density. These topologies minimize communication overhead."
        )
    
    # Rule 6: General recommendation
    if 'star' in configs and 'cascade' in configs:
        star_acc = accuracies.get('star', 0)
        cascade_acc = accuracies.get('cascade', 0)
        
        if star_acc > cascade_acc * 1.1:
            rules.append(
                f"🌟 **Hub Architecture**: Star topology significantly outperforms sequential cascade ({star_acc:.3f} vs {cascade_acc:.3f}). "
                f"Centralized coordination is beneficial for this problem."
            )
        elif cascade_acc > star_acc * 1.1:
            rules.append(
                f"🔗 **Sequential Processing**: Cascade topology outperforms centralized star ({cascade_acc:.3f} vs {star_acc:.3f}). "
                f"Sequential information flow is more effective for this task."
            )
    
    return rules


def generate_report(all_results: Dict[str, Dict[str, Any]], 
                   config: Dict[str, Any],
                   output_path: str):
    """
    Generate comprehensive markdown report
    
    Args:
        all_results: Dictionary with all experimental results
        config: Configuration dictionary
        output_path: Path to save report
    """
    report = []
    
    # Header
    report.append("# Multi-Agent AI Systems Benchmark Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("\nThis report presents a comprehensive evaluation of multi-agent AI systems ")
    report.append("across different network topologies. The benchmark assesses accuracy, robustness, ")
    report.append("error propagation, and failure resilience.\n")
    
    # Configuration
    report.append("## Experimental Configuration\n")
    report.append(f"- **Dataset**: {config.get('dataset', 'N/A')}")
    report.append(f"- **Agent Model**: {config.get('agent_type', 'N/A')}")
    report.append(f"- **Number of Agents**: {config.get('n_agents', 'N/A')}")
    report.append(f"- **Communication Rounds**: {config.get('n_rounds', 'N/A')}")
    report.append(f"- **Topologies Tested**: {', '.join(all_results.keys())}")
    report.append("")
    
    # Results Summary Table
    report.append("## Results Summary\n")
    report.append("| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |")
    report.append("|----------|----------------|--------------|------------|-------------|-----------------|")
    
    for config_name, results in all_results.items():
        final_acc = results.get('final_accuracy', 0)
        avg_acc = results.get('average_accuracy', 0)
        robust = results.get('robustness', 0)
        error_depth = results.get('error_depth', {}).get('mean_depth', 0)
        density = results.get('network_info', {}).get('density', 0)
        
        report.append(f"| {config_name} | {final_acc:.4f} | {avg_acc:.4f} | "
                     f"{robust:.4f} | {error_depth:.4f} | {density:.4f} |")
    
    report.append("")
    
    # Detailed Analysis
    report.append("## Detailed Analysis\n")
    
    for config_name, results in all_results.items():
        report.append(f"### {config_name}\n")
        
        report.append(f"**Performance Metrics:**")
        report.append(f"- Final Accuracy: {results.get('final_accuracy', 0):.4f}")
        report.append(f"- Average Accuracy: {results.get('average_accuracy', 0):.4f}")
        report.append(f"- Accuracy Std Dev: {results.get('accuracy_std', 0):.4f}")
        report.append(f"- Robustness Score: {results.get('robustness', 0):.4f}")
        
        report.append(f"\n**Network Properties:**")
        net_info = results.get('network_info', {})
        report.append(f"- Number of Agents: {net_info.get('n_agents', 0)}")
        report.append(f"- Number of Edges: {net_info.get('n_edges', 0)}")
        report.append(f"- Network Density: {net_info.get('density', 0):.4f}")
        
        report.append(f"\n**Reliability:**")
        report.append(f"- Max Failed Nodes: {results.get('max_failed_nodes', 0)}")
        report.append(f"- Total Messages Exchanged: {results.get('total_messages', 0)}")
        
        error_depth = results.get('error_depth', {})
        if error_depth:
            report.append(f"\n**Error Propagation:**")
            report.append(f"- Mean Error Depth: {error_depth.get('mean_depth', 0):.4f}")
            report.append(f"- Max Error Depth: {error_depth.get('max_depth', 0):.4f}")
            report.append(f"- Error Rate: {error_depth.get('error_rate', 0):.4f}")
        
        centrality = results.get('failed_node_centrality', {})
        if centrality:
            report.append(f"\n**Failed Node Impact:**")
            report.append(f"- Avg Degree Centrality: {centrality.get('avg_degree_centrality', 0):.4f}")
            report.append(f"- Avg Betweenness Centrality: {centrality.get('avg_betweenness_centrality', 0):.4f}")
            report.append(f"- Failure Rate: {centrality.get('failure_rate', 0):.4f}")
        
        report.append("")
    
    # Design Rules
    report.append("## Design Rules & Recommendations\n")
    report.append("Based on the experimental results, we derive the following design guidelines:\n")
    
    design_rules = generate_design_rules(all_results)
    for i, rule in enumerate(design_rules, 1):
        report.append(f"{i}. {rule}\n")
    
    # Conclusion
    report.append("## Conclusion\n")
    
    best_config = max(all_results.keys(), 
                     key=lambda c: all_results[c].get('final_accuracy', 0))
    best_accuracy = all_results[best_config].get('final_accuracy', 0)
    
    report.append(f"The **{best_config}** topology achieved the highest final accuracy of {best_accuracy:.4f}. ")
    report.append("However, the optimal choice depends on the specific application requirements:\n")
    report.append("- For maximum accuracy: Choose the topology with highest final accuracy")
    report.append("- For reliability: Choose the topology with highest robustness score")
    report.append("- For efficiency: Choose topologies with low network density but high accuracy")
    report.append("- For scalability: Consider topologies that maintain performance as network size increases")
    
    report.append("\n---")
    report.append("\n*Report generated by Multi-Agent AI Systems Benchmarking Framework*")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated: {output_path}")

