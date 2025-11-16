# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-11-16 16:31:18

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
- **Topologies Tested**: star_logistic, cascade_logistic, feedback_rewired_logistic, ring_logistic, mesh_logistic, small_world_logistic, scale_free_logistic, tree_logistic, grid_logistic, star_neural, cascade_neural, feedback_rewired_neural, ring_neural, mesh_neural, small_world_neural, scale_free_neural, tree_neural, grid_neural, star_gat, cascade_gat, feedback_rewired_gat, ring_gat, mesh_gat, small_world_gat, scale_free_gat, tree_gat, grid_gat, star_rf, cascade_rf, feedback_rewired_rf, ring_rf, mesh_rf, small_world_rf, scale_free_rf, tree_rf, grid_rf

## Results Summary

| Topology | Final Accuracy | Avg Accuracy | Robustness | Error Depth | Network Density |
|----------|----------------|--------------|------------|-------------|-----------------|
| star_logistic | 0.6802 | 0.6767 | 0.9998 | 1.6200 | 0.2000 |
| cascade_logistic | 0.6310 | 0.6376 | 0.9999 | 2.2500 | 0.1000 |
| feedback_rewired_logistic | 0.6396 | 0.6512 | 0.9999 | 2.2392 | 0.1556 |
| ring_logistic | 0.6728 | 0.6741 | 0.9998 | 2.5000 | 0.2222 |
| mesh_logistic | 0.6716 | 0.6696 | 0.9999 | 0.9000 | 1.0000 |
| small_world_logistic | 0.6777 | 0.6748 | 0.9999 | 1.4200 | 0.4444 |
| scale_free_logistic | 0.6765 | 0.6748 | 0.9999 | 1.4000 | 0.4667 |
| tree_logistic | 0.6691 | 0.6660 | 0.9999 | 2.2200 | 0.2000 |
| grid_logistic | 0.6740 | 0.6700 | 0.9999 | 2.1000 | 0.2889 |
| star_neural | 0.1550 | 0.1624 | 0.9989 | 1.6200 | 0.2000 |
| cascade_neural | 0.2977 | 0.2539 | 0.9983 | 2.2500 | 0.1000 |
| feedback_rewired_neural | 0.2694 | 0.2608 | 0.9996 | 1.7032 | 0.1222 |
| ring_neural | 0.1550 | 0.1618 | 0.9992 | 2.5000 | 0.2222 |
| mesh_neural | 0.1550 | 0.1598 | 0.9993 | 0.9000 | 1.0000 |
| small_world_neural | 0.1550 | 0.1598 | 0.9991 | 1.4200 | 0.4444 |
| scale_free_neural | 0.1550 | 0.1603 | 0.9992 | 1.4000 | 0.4667 |
| tree_neural | 0.1550 | 0.1646 | 0.9991 | 2.2200 | 0.2000 |
| grid_neural | 0.1550 | 0.1614 | 0.9990 | 2.1000 | 0.2889 |
| star_gat | 0.1095 | 0.1095 | 1.0000 | 1.6200 | 0.2000 |
| cascade_gat | 0.1095 | 0.1095 | 1.0000 | 2.2500 | 0.1000 |
| feedback_rewired_gat | 0.1095 | 0.1095 | 1.0000 | 3.2011 | 0.1556 |
| ring_gat | 0.1095 | 0.1095 | 1.0000 | 2.5000 | 0.2222 |
| mesh_gat | 0.1095 | 0.1095 | 1.0000 | 0.9000 | 1.0000 |
| small_world_gat | 0.1095 | 0.1095 | 1.0000 | 1.4200 | 0.4444 |
| scale_free_gat | 0.1095 | 0.1095 | 1.0000 | 1.4000 | 0.4667 |
| tree_gat | 0.1095 | 0.1095 | 1.0000 | 2.2200 | 0.2000 |
| grid_gat | 0.1095 | 0.1095 | 1.0000 | 2.1000 | 0.2889 |
| star_rf | 0.5523 | 0.5523 | 1.0000 | 1.6200 | 0.2000 |
| cascade_rf | 0.5523 | 0.5523 | 1.0000 | 2.2500 | 0.1000 |
| feedback_rewired_rf | 0.5523 | 0.5523 | 1.0000 | 3.8944 | 0.1333 |
| ring_rf | 0.5523 | 0.5523 | 1.0000 | 2.5000 | 0.2222 |
| mesh_rf | 0.5523 | 0.5523 | 1.0000 | 0.9000 | 1.0000 |
| small_world_rf | 0.5523 | 0.5523 | 1.0000 | 1.4200 | 0.4444 |
| scale_free_rf | 0.5523 | 0.5523 | 1.0000 | 1.4000 | 0.4667 |
| tree_rf | 0.5523 | 0.5523 | 1.0000 | 2.2200 | 0.2000 |
| grid_rf | 0.5523 | 0.5523 | 1.0000 | 2.1000 | 0.2889 |

## Detailed Analysis

### star_logistic

**Performance Metrics:**
- Final Accuracy: 0.6802
- Average Accuracy: 0.6767
- Accuracy Std Dev: 0.0130
- Robustness Score: 0.9998

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 1.6200
- Max Error Depth: 1.7000
- Error Rate: 0.3198

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### cascade_logistic

**Performance Metrics:**
- Final Accuracy: 0.6310
- Average Accuracy: 0.6376
- Accuracy Std Dev: 0.0089
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 270

**Error Propagation:**
- Mean Error Depth: 2.2500
- Max Error Depth: 4.5000
- Error Rate: 0.3690

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### feedback_rewired_logistic

**Performance Metrics:**
- Final Accuracy: 0.6396
- Average Accuracy: 0.6512
- Accuracy Std Dev: 0.0105
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 14
- Network Density: 0.1556

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 412

**Error Propagation:**
- Mean Error Depth: 2.2392
- Max Error Depth: 4.3000
- Error Rate: 0.3604

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### ring_logistic

**Performance Metrics:**
- Final Accuracy: 0.6728
- Average Accuracy: 0.6741
- Accuracy Std Dev: 0.0123
- Robustness Score: 0.9998

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 600

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.3284

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### mesh_logistic

**Performance Metrics:**
- Final Accuracy: 0.6716
- Average Accuracy: 0.6696
- Accuracy Std Dev: 0.0109
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 2700

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.3284

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### small_world_logistic

**Performance Metrics:**
- Final Accuracy: 0.6777
- Average Accuracy: 0.6748
- Accuracy Std Dev: 0.0119
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1200

**Error Propagation:**
- Mean Error Depth: 1.4200
- Max Error Depth: 1.7000
- Error Rate: 0.3223

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### scale_free_logistic

**Performance Metrics:**
- Final Accuracy: 0.6765
- Average Accuracy: 0.6748
- Accuracy Std Dev: 0.0119
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1260

**Error Propagation:**
- Mean Error Depth: 1.4000
- Max Error Depth: 1.7000
- Error Rate: 0.3235

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### tree_logistic

**Performance Metrics:**
- Final Accuracy: 0.6691
- Average Accuracy: 0.6660
- Accuracy Std Dev: 0.0117
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 2.2200
- Max Error Depth: 2.5000
- Error Rate: 0.3284

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### grid_logistic

**Performance Metrics:**
- Final Accuracy: 0.6740
- Average Accuracy: 0.6700
- Accuracy Std Dev: 0.0111
- Robustness Score: 0.9999

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 780

**Error Propagation:**
- Mean Error Depth: 2.1000
- Max Error Depth: 2.5000
- Error Rate: 0.3284

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### star_neural

**Performance Metrics:**
- Final Accuracy: 0.1550
- Average Accuracy: 0.1624
- Accuracy Std Dev: 0.0337
- Robustness Score: 0.9989

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 1.6200
- Max Error Depth: 1.7000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### cascade_neural

**Performance Metrics:**
- Final Accuracy: 0.2977
- Average Accuracy: 0.2539
- Accuracy Std Dev: 0.0409
- Robustness Score: 0.9983

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 270

**Error Propagation:**
- Mean Error Depth: 2.2500
- Max Error Depth: 4.5000
- Error Rate: 0.7023

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### feedback_rewired_neural

**Performance Metrics:**
- Final Accuracy: 0.2694
- Average Accuracy: 0.2608
- Accuracy Std Dev: 0.0199
- Robustness Score: 0.9996

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 11
- Network Density: 0.1222

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 327

**Error Propagation:**
- Mean Error Depth: 1.7032
- Max Error Depth: 3.3750
- Error Rate: 0.6568

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### ring_neural

**Performance Metrics:**
- Final Accuracy: 0.1550
- Average Accuracy: 0.1618
- Accuracy Std Dev: 0.0281
- Robustness Score: 0.9992

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 600

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### mesh_neural

**Performance Metrics:**
- Final Accuracy: 0.1550
- Average Accuracy: 0.1598
- Accuracy Std Dev: 0.0265
- Robustness Score: 0.9993

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 2700

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### small_world_neural

**Performance Metrics:**
- Final Accuracy: 0.1550
- Average Accuracy: 0.1598
- Accuracy Std Dev: 0.0295
- Robustness Score: 0.9991

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1200

**Error Propagation:**
- Mean Error Depth: 1.4200
- Max Error Depth: 1.7000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### scale_free_neural

**Performance Metrics:**
- Final Accuracy: 0.1550
- Average Accuracy: 0.1603
- Accuracy Std Dev: 0.0278
- Robustness Score: 0.9992

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1260

**Error Propagation:**
- Mean Error Depth: 1.4000
- Max Error Depth: 1.7000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### tree_neural

**Performance Metrics:**
- Final Accuracy: 0.1550
- Average Accuracy: 0.1646
- Accuracy Std Dev: 0.0303
- Robustness Score: 0.9991

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 2.2200
- Max Error Depth: 2.5000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### grid_neural

**Performance Metrics:**
- Final Accuracy: 0.1550
- Average Accuracy: 0.1614
- Accuracy Std Dev: 0.0319
- Robustness Score: 0.9990

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 780

**Error Propagation:**
- Mean Error Depth: 2.1000
- Max Error Depth: 2.5000
- Error Rate: 0.8450

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### star_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 1.6200
- Max Error Depth: 1.7000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### cascade_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 270

**Error Propagation:**
- Mean Error Depth: 2.2500
- Max Error Depth: 4.5000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### feedback_rewired_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 14
- Network Density: 0.1556

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 413

**Error Propagation:**
- Mean Error Depth: 3.2011
- Max Error Depth: 3.8889
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### ring_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 600

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### mesh_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 2700

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### small_world_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1200

**Error Propagation:**
- Mean Error Depth: 1.4200
- Max Error Depth: 1.7000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### scale_free_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1260

**Error Propagation:**
- Mean Error Depth: 1.4000
- Max Error Depth: 1.7000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### tree_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 2.2200
- Max Error Depth: 2.5000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### grid_gat

**Performance Metrics:**
- Final Accuracy: 0.1095
- Average Accuracy: 0.1095
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 780

**Error Propagation:**
- Mean Error Depth: 2.1000
- Max Error Depth: 2.5000
- Error Rate: 0.8905

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### star_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 1.6200
- Max Error Depth: 1.7000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### cascade_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 270

**Error Propagation:**
- Mean Error Depth: 2.2500
- Max Error Depth: 4.5000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### feedback_rewired_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 12
- Network Density: 0.1333

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 355

**Error Propagation:**
- Mean Error Depth: 3.8944
- Max Error Depth: 4.5000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### ring_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 600

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### mesh_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 2700

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### small_world_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1200

**Error Propagation:**
- Mean Error Depth: 1.4200
- Max Error Depth: 1.7000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### scale_free_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 1260

**Error Propagation:**
- Mean Error Depth: 1.4000
- Max Error Depth: 1.7000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### tree_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 540

**Error Propagation:**
- Mean Error Depth: 2.2200
- Max Error Depth: 2.5000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

### grid_rf

**Performance Metrics:**
- Final Accuracy: 0.5523
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0000
- Robustness Score: 1.0000

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 0
- Total Messages Exchanged: 780

**Error Propagation:**
- Mean Error Depth: 2.1000
- Max Error Depth: 2.5000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.0000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: star_logistic achieved 0.680 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: star_gat achieved 1.000 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. ⚡ **Communication Efficient**: star_logistic, cascade_logistic, feedback_rewired_logistic, ring_logistic, tree_logistic, grid_logistic, star_rf, cascade_rf, feedback_rewired_rf, ring_rf, tree_rf, grid_rf achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **star_logistic** topology achieved the highest final accuracy of 0.6802. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*