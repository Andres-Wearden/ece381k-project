# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-11-16 17:07:55

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
- **Topologies Tested**: star_topology_logistic, cascade_topology_logistic, feedback_rewired_topology_logistic, mesh_topology_logistic, scale_free_topology_logistic

## Results Summary

| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |
|----------|----------------|--------------|------------|-------------|-----------------|
| star_topology_logistic | 0.2964 | 0.3122 | 0.9888 | 1.9090 | 0.0400 |
| cascade_topology_logistic | 0.3652 | 0.3833 | 0.9973 | 13.4787 | 0.0514 |
| feedback_rewired_topology_logistic | 0.4009 | 0.3690 | 0.9819 | 5.7526 | 0.0290 |
| mesh_topology_logistic | 0.3447 | 0.3549 | 0.9916 | 0.9800 | 1.0000 |
| scale_free_topology_logistic | 0.3375 | 0.3348 | 0.9828 | 2.4267 | 0.0784 |

## Detailed Analysis

### star_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.2964
- Average Accuracy: 0.3122
- Accuracy Std Dev: 0.0169
- Robustness Score: 0.9888

**Network Properties:**
- Number of Agents: 50
- Number of Edges: 98
- Network Density: 0.0400

**Reliability:**
- Max Failed Nodes: 23
- Total Messages Exchanged: 2246

**Error Propagation:**
- Mean Error Depth: 1.9090
- Max Error Depth: 1.9400
- Error Rate: 0.6863

**Failed Node Impact:**
- Avg Degree Centrality: 0.0204
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.3800

### cascade_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.3652
- Average Accuracy: 0.3833
- Accuracy Std Dev: 0.0165
- Robustness Score: 0.9973

**Network Properties:**
- Number of Agents: 50
- Number of Edges: 126
- Network Density: 0.0514

**Reliability:**
- Max Failed Nodes: 25
- Total Messages Exchanged: 2304

**Error Propagation:**
- Mean Error Depth: 13.4787
- Max Error Depth: 23.5400
- Error Rate: 0.6101

**Failed Node Impact:**
- Avg Degree Centrality: 0.0567
- Avg Betweenness Centrality: 0.2550
- Failure Rate: 0.3600

### feedback_rewired_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.4009
- Average Accuracy: 0.3690
- Accuracy Std Dev: 0.0177
- Robustness Score: 0.9819

**Network Properties:**
- Number of Agents: 50
- Number of Edges: 71
- Network Density: 0.0290

**Reliability:**
- Max Failed Nodes: 24
- Total Messages Exchanged: 1304

**Error Propagation:**
- Mean Error Depth: 5.7526
- Max Error Depth: 9.5000
- Error Rate: 0.6396

**Failed Node Impact:**
- Avg Degree Centrality: 0.0447
- Avg Betweenness Centrality: 0.0571
- Failure Rate: 0.4200

### mesh_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.3447
- Average Accuracy: 0.3549
- Accuracy Std Dev: 0.0107
- Robustness Score: 0.9916

**Network Properties:**
- Number of Agents: 50
- Number of Edges: 2450
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 21
- Total Messages Exchanged: 45880

**Error Propagation:**
- Mean Error Depth: 0.9800
- Max Error Depth: 0.9800
- Error Rate: 0.6371

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.3400

### scale_free_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.3375
- Average Accuracy: 0.3348
- Accuracy Std Dev: 0.0165
- Robustness Score: 0.9828

**Network Properties:**
- Number of Agents: 50
- Number of Edges: 192
- Network Density: 0.0784

**Reliability:**
- Max Failed Nodes: 18
- Total Messages Exchanged: 3954

**Error Propagation:**
- Mean Error Depth: 2.4267
- Max Error Depth: 3.0400
- Error Rate: 0.6679

**Failed Node Impact:**
- Avg Degree Centrality: 0.0594
- Avg Betweenness Centrality: 0.0087
- Failure Rate: 0.2200

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: feedback_rewired_topology_logistic achieved 0.401 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: cascade_topology_logistic achieved 0.997 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: star_topology_logistic maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: cascade_topology_logistic, feedback_rewired_topology_logistic achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **feedback_rewired_topology_logistic** topology achieved the highest final accuracy of 0.4009. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*