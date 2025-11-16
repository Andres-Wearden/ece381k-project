# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-11-16 17:10:49

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
| star_topology_logistic | 0.6851 | 0.6842 | 0.9996 | 1.6000 | 0.2000 |
| cascade_topology_logistic | 0.6544 | 0.6631 | 0.9966 | 2.7333 | 0.2667 |
| feedback_rewired_topology_logistic | 0.6299 | 0.6611 | 0.9997 | 4.2667 | 0.1444 |
| mesh_topology_logistic | 0.6636 | 0.6763 | 0.9995 | 0.9000 | 1.0000 |
| scale_free_topology_logistic | 0.6877 | 0.6801 | 0.9996 | 1.5714 | 0.3556 |

## Detailed Analysis

### star_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6851
- Average Accuracy: 0.6842
- Accuracy Std Dev: 0.0206
- Robustness Score: 0.9996

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 354

**Error Propagation:**
- Mean Error Depth: 1.6000
- Max Error Depth: 1.7000
- Error Rate: 0.3198

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.2000

### cascade_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6544
- Average Accuracy: 0.6631
- Accuracy Std Dev: 0.0191
- Robustness Score: 0.9966

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 24
- Network Density: 0.2667

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 440

**Error Propagation:**
- Mean Error Depth: 2.7333
- Max Error Depth: 4.5000
- Error Rate: 0.3383

**Failed Node Impact:**
- Avg Degree Centrality: 0.3333
- Avg Betweenness Centrality: 0.2297
- Failure Rate: 0.4000

### feedback_rewired_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6299
- Average Accuracy: 0.6611
- Accuracy Std Dev: 0.0179
- Robustness Score: 0.9997

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 13
- Network Density: 0.1444

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 235

**Error Propagation:**
- Mean Error Depth: 4.2667
- Max Error Depth: 4.5000
- Error Rate: 0.3296

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.4688
- Failure Rate: 0.4000

### mesh_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6636
- Average Accuracy: 0.6763
- Accuracy Std Dev: 0.0231
- Robustness Score: 0.9995

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 8
- Total Messages Exchanged: 1488

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.3284

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.2000

### scale_free_topology_logistic

**Performance Metrics:**
- Final Accuracy: 0.6877
- Average Accuracy: 0.6801
- Accuracy Std Dev: 0.0203
- Robustness Score: 0.9996

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 32
- Network Density: 0.3556

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 586

**Error Propagation:**
- Mean Error Depth: 1.5714
- Max Error Depth: 1.9000
- Error Rate: 0.3260

**Failed Node Impact:**
- Avg Degree Centrality: 0.4074
- Avg Betweenness Centrality: 0.1358
- Failure Rate: 0.3000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: scale_free_topology_logistic achieved 0.688 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: feedback_rewired_topology_logistic achieved 1.000 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: mesh_topology_logistic maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: star_topology_logistic, scale_free_topology_logistic achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **scale_free_topology_logistic** topology achieved the highest final accuracy of 0.6877. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*