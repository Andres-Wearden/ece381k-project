# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-12-08 17:26:50

---

## Executive Summary

This report presents a comprehensive evaluation of multi-agent AI systems 
across different network topologies. The benchmark assesses accuracy, robustness, 
error propagation, and failure resilience.

## Experimental Configuration

- **Dataset**: N/A
- **Agent Model**: N/A
- **Number of Agents**: N/A
- **Communication Rounds**: N/A
- **Topologies Tested**: star_topology_neural, cascade_topology_neural, feedback_rewired_topology_neural, mesh_topology_neural, scale_free_topology_neural

## Results Summary

| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |
|----------|----------------|--------------|------------|-------------|-----------------|
| star_topology_neural | 0.4020 | 0.3774 | 0.9740 | 1.5400 | 0.2000 |
| cascade_topology_neural | 0.2966 | 0.3094 | 0.9699 | 2.9000 | 0.2556 |
| feedback_rewired_topology_neural | 0.4753 | 0.4559 | 0.9956 | 2.5417 | 0.1444 |
| mesh_topology_neural | 0.3039 | 0.3105 | 0.9786 | 0.9000 | 1.0000 |
| scale_free_topology_neural | 0.3210 | 0.3094 | 0.9712 | 1.5250 | 0.3556 |

## Detailed Analysis

### star_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.4020
- Average Accuracy: 0.3774
- Accuracy Std Dev: 0.0339
- Robustness Score: 0.9740

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 356

**Error Propagation:**
- Mean Error Depth: 1.5400
- Max Error Depth: 1.7000
- Error Rate: 0.6298

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.5000

### cascade_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.2966
- Average Accuracy: 0.3094
- Accuracy Std Dev: 0.0452
- Robustness Score: 0.9699

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 23
- Network Density: 0.2556

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 515

**Error Propagation:**
- Mean Error Depth: 2.9000
- Max Error Depth: 4.5000
- Error Rate: 0.6974

**Failed Node Impact:**
- Avg Degree Centrality: 0.2889
- Avg Betweenness Centrality: 0.1958
- Failure Rate: 0.5000

### feedback_rewired_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.4753
- Average Accuracy: 0.4559
- Accuracy Std Dev: 0.0664
- Robustness Score: 0.9956

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 13
- Network Density: 0.1444

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 240

**Error Propagation:**
- Mean Error Depth: 2.5417
- Max Error Depth: 4.0000
- Error Rate: 0.5437

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.1829
- Failure Rate: 0.6000

### mesh_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.3039
- Average Accuracy: 0.3105
- Accuracy Std Dev: 0.0424
- Robustness Score: 0.9786

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 2120

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.6974

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.5000

### scale_free_topology_neural

**Performance Metrics:**
- Final Accuracy: 0.3210
- Average Accuracy: 0.3094
- Accuracy Std Dev: 0.0443
- Robustness Score: 0.9712

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 618

**Error Propagation:**
- Mean Error Depth: 1.5250
- Max Error Depth: 1.9000
- Error Rate: 0.6974

**Failed Node Impact:**
- Avg Degree Centrality: 0.3333
- Avg Betweenness Centrality: 0.0826
- Failure Rate: 0.6000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: feedback_rewired_topology_neural achieved 0.475 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: feedback_rewired_topology_neural achieved 0.996 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: scale_free_topology_neural maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: star_topology_neural, feedback_rewired_topology_neural achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **feedback_rewired_topology_neural** topology achieved the highest final accuracy of 0.4753. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*