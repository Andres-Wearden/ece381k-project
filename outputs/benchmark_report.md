# Multi-Agent AI Systems Benchmark Report

**Generated**: 2025-12-08 20:11:43

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
| star_logistic | 0.6759 | 0.6695 | 0.4656 | 1.6000 | 0.2000 |
| cascade_logistic | 0.6544 | 0.6376 | 0.3287 | 1.2000 | 0.1000 |
| feedback_rewired_logistic | 0.6366 | 0.6392 | 0.3708 | 2.0370 | 0.1556 |
| ring_logistic | 0.6789 | 0.6701 | 0.4147 | 2.5000 | 0.2222 |
| mesh_logistic | 0.6776 | 0.6671 | 0.7268 | 0.9000 | 1.0000 |
| small_world_logistic | 0.7136 | 0.6756 | 0.5207 | 1.4200 | 0.4444 |
| scale_free_logistic | 0.6708 | 0.6717 | 0.5484 | 1.3500 | 0.4667 |
| tree_logistic | 0.6748 | 0.6631 | 0.4187 | 2.3000 | 0.2000 |
| grid_logistic | 0.6698 | 0.6682 | 0.4589 | 2.0500 | 0.2889 |
| star_neural | 0.2724 | 0.2898 | 0.4170 | 1.4333 | 0.2000 |
| cascade_neural | 0.2699 | 0.2631 | 0.3211 | 1.9167 | 0.1000 |
| feedback_rewired_neural | 0.2840 | 0.2611 | 0.3507 | 1.8750 | 0.1333 |
| ring_neural | 0.2778 | 0.2750 | 0.3897 | 2.5000 | 0.2222 |
| mesh_neural | 0.2527 | 0.2610 | 0.6931 | 0.9000 | 1.0000 |
| small_world_neural | 0.2857 | 0.2832 | 0.5213 | 1.4429 | 0.4444 |
| scale_free_neural | 0.2822 | 0.2880 | 0.5135 | 1.3833 | 0.4667 |
| tree_neural | 0.3180 | 0.2730 | 0.4110 | 2.2500 | 0.2000 |
| grid_neural | 0.3235 | 0.2996 | 0.4213 | 2.3800 | 0.2889 |
| star_gat | 0.6343 | 0.6392 | 0.4510 | 1.7000 | 0.2000 |
| cascade_gat | 0.6933 | 0.6328 | 0.3220 | 1.6667 | 0.1000 |
| feedback_rewired_gat | 0.5767 | 0.6248 | 0.3414 | 2.6913 | 0.1667 |
| ring_gat | 0.6933 | 0.6893 | 0.4057 | 2.5000 | 0.2222 |
| mesh_gat | 0.7132 | 0.6898 | 0.7306 | 0.9000 | 1.0000 |
| small_world_gat | 0.7254 | 0.7187 | 0.5258 | 1.4333 | 0.4444 |
| scale_free_gat | 0.6851 | 0.6753 | 0.5185 | 1.4667 | 0.4667 |
| tree_gat | 0.6528 | 0.6591 | 0.4273 | 2.1500 | 0.2000 |
| grid_gat | 0.7593 | 0.7360 | 0.4422 | 1.9667 | 0.2889 |
| star_rf | 0.5193 | 0.5438 | 0.4269 | 1.7000 | 0.2000 |
| cascade_rf | 0.5438 | 0.5523 | 0.3158 | 2.0000 | 0.1000 |
| feedback_rewired_rf | 0.5545 | 0.5469 | 0.3672 | 2.0417 | 0.1556 |
| ring_rf | 0.5453 | 0.5400 | 0.3775 | 2.5000 | 0.2222 |
| mesh_rf | 0.4954 | 0.5509 | 0.7038 | 0.9000 | 1.0000 |
| small_world_rf | 0.5281 | 0.5465 | 0.4992 | 1.4143 | 0.4444 |
| scale_free_rf | 0.5649 | 0.5625 | 0.5282 | 1.4429 | 0.4667 |
| tree_rf | 0.5484 | 0.5554 | 0.4105 | 2.3750 | 0.2000 |
| grid_rf | 0.5316 | 0.5563 | 0.4499 | 1.9286 | 0.2889 |

## Detailed Analysis

### star_logistic

**Performance Metrics:**
- Final Accuracy: 0.6759
- Average Accuracy: 0.6695
- Accuracy Std Dev: 0.0176
- Robustness Score: 0.4656

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 9
- Total Messages Exchanged: 308

**Error Propagation:**
- Mean Error Depth: 1.6000
- Max Error Depth: 1.7000
- Error Rate: 0.3198

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.2000

### cascade_logistic

**Performance Metrics:**
- Final Accuracy: 0.6544
- Average Accuracy: 0.6376
- Accuracy Std Dev: 0.0143
- Robustness Score: 0.3287

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 151

**Error Propagation:**
- Mean Error Depth: 1.2000
- Max Error Depth: 3.0000
- Error Rate: 0.3309

**Failed Node Impact:**
- Avg Degree Centrality: 0.2000
- Avg Betweenness Centrality: 0.1722
- Failure Rate: 0.5000

### feedback_rewired_logistic

**Performance Metrics:**
- Final Accuracy: 0.6366
- Average Accuracy: 0.6392
- Accuracy Std Dev: 0.0136
- Robustness Score: 0.3708

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 14
- Network Density: 0.1556

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 243

**Error Propagation:**
- Mean Error Depth: 2.0370
- Max Error Depth: 3.8000
- Error Rate: 0.3567

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.1000

### ring_logistic

**Performance Metrics:**
- Final Accuracy: 0.6789
- Average Accuracy: 0.6701
- Accuracy Std Dev: 0.0193
- Robustness Score: 0.4147

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 328

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.3272

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.2222
- Failure Rate: 0.3000

### mesh_logistic

**Performance Metrics:**
- Final Accuracy: 0.6776
- Average Accuracy: 0.6671
- Accuracy Std Dev: 0.0174
- Robustness Score: 0.7268

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 1700

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.3284

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.1000

### small_world_logistic

**Performance Metrics:**
- Final Accuracy: 0.7136
- Average Accuracy: 0.6756
- Accuracy Std Dev: 0.0261
- Robustness Score: 0.5207

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 544

**Error Propagation:**
- Mean Error Depth: 1.4200
- Max Error Depth: 1.7000
- Error Rate: 0.3235

**Failed Node Impact:**
- Avg Degree Centrality: 0.4222
- Avg Betweenness Centrality: 0.0546
- Failure Rate: 0.5000

### scale_free_logistic

**Performance Metrics:**
- Final Accuracy: 0.6708
- Average Accuracy: 0.6717
- Accuracy Std Dev: 0.0154
- Robustness Score: 0.5484

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 740

**Error Propagation:**
- Mean Error Depth: 1.3500
- Max Error Depth: 1.7000
- Error Rate: 0.3235

**Failed Node Impact:**
- Avg Degree Centrality: 0.3889
- Avg Betweenness Centrality: 0.0359
- Failure Rate: 0.4000

### tree_logistic

**Performance Metrics:**
- Final Accuracy: 0.6748
- Average Accuracy: 0.6631
- Accuracy Std Dev: 0.0273
- Robustness Score: 0.4187

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 8
- Total Messages Exchanged: 276

**Error Propagation:**
- Mean Error Depth: 2.3000
- Max Error Depth: 2.5000
- Error Rate: 0.3272

**Failed Node Impact:**
- Avg Degree Centrality: 0.2778
- Avg Betweenness Centrality: 0.2917
- Failure Rate: 0.4000

### grid_logistic

**Performance Metrics:**
- Final Accuracy: 0.6698
- Average Accuracy: 0.6682
- Accuracy Std Dev: 0.0154
- Robustness Score: 0.4589

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 444

**Error Propagation:**
- Mean Error Depth: 2.0500
- Max Error Depth: 2.5000
- Error Rate: 0.3296

**Failed Node Impact:**
- Avg Degree Centrality: 0.2778
- Avg Betweenness Centrality: 0.1556
- Failure Rate: 0.6000

### star_neural

**Performance Metrics:**
- Final Accuracy: 0.2724
- Average Accuracy: 0.2898
- Accuracy Std Dev: 0.0173
- Robustness Score: 0.4170

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 174

**Error Propagation:**
- Mean Error Depth: 1.4333
- Max Error Depth: 1.7000
- Error Rate: 0.6974

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.7000

### cascade_neural

**Performance Metrics:**
- Final Accuracy: 0.2699
- Average Accuracy: 0.2631
- Accuracy Std Dev: 0.0155
- Robustness Score: 0.3211

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 145

**Error Propagation:**
- Mean Error Depth: 1.9167
- Max Error Depth: 4.5000
- Error Rate: 0.6876

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.2083
- Failure Rate: 0.4000

### feedback_rewired_neural

**Performance Metrics:**
- Final Accuracy: 0.2840
- Average Accuracy: 0.2611
- Accuracy Std Dev: 0.0231
- Robustness Score: 0.3507

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 12
- Network Density: 0.1333

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 170

**Error Propagation:**
- Mean Error Depth: 1.8750
- Max Error Depth: 2.8750
- Error Rate: 0.6396

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.0694
- Failure Rate: 0.4000

### ring_neural

**Performance Metrics:**
- Final Accuracy: 0.2778
- Average Accuracy: 0.2750
- Accuracy Std Dev: 0.0204
- Robustness Score: 0.3897

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 316

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.7060

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.2222
- Failure Rate: 0.6000

### mesh_neural

**Performance Metrics:**
- Final Accuracy: 0.2527
- Average Accuracy: 0.2610
- Accuracy Std Dev: 0.0147
- Robustness Score: 0.6931

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 1306

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.6863

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.1000

### small_world_neural

**Performance Metrics:**
- Final Accuracy: 0.2857
- Average Accuracy: 0.2832
- Accuracy Std Dev: 0.0188
- Robustness Score: 0.5213

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 472

**Error Propagation:**
- Mean Error Depth: 1.4429
- Max Error Depth: 1.7000
- Error Rate: 0.6900

**Failed Node Impact:**
- Avg Degree Centrality: 0.4815
- Avg Betweenness Centrality: 0.0864
- Failure Rate: 0.3000

### scale_free_neural

**Performance Metrics:**
- Final Accuracy: 0.2822
- Average Accuracy: 0.2880
- Accuracy Std Dev: 0.0211
- Robustness Score: 0.5135

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 8
- Total Messages Exchanged: 664

**Error Propagation:**
- Mean Error Depth: 1.3833
- Max Error Depth: 1.5000
- Error Rate: 0.6814

**Failed Node Impact:**
- Avg Degree Centrality: 0.4722
- Avg Betweenness Centrality: 0.0822
- Failure Rate: 0.4000

### tree_neural

**Performance Metrics:**
- Final Accuracy: 0.3180
- Average Accuracy: 0.2730
- Accuracy Std Dev: 0.0209
- Robustness Score: 0.4110

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 308

**Error Propagation:**
- Mean Error Depth: 2.2500
- Max Error Depth: 2.5000
- Error Rate: 0.6900

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.1944
- Failure Rate: 0.6000

### grid_neural

**Performance Metrics:**
- Final Accuracy: 0.3235
- Average Accuracy: 0.2996
- Accuracy Std Dev: 0.0219
- Robustness Score: 0.4213

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 432

**Error Propagation:**
- Mean Error Depth: 2.3800
- Max Error Depth: 2.5000
- Error Rate: 0.6913

**Failed Node Impact:**
- Avg Degree Centrality: 0.3333
- Avg Betweenness Centrality: 0.2583
- Failure Rate: 0.5000

### star_gat

**Performance Metrics:**
- Final Accuracy: 0.6343
- Average Accuracy: 0.6392
- Accuracy Std Dev: 0.0201
- Robustness Score: 0.4510

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 280

**Error Propagation:**
- Mean Error Depth: 1.7000
- Max Error Depth: 1.7000
- Error Rate: 0.3764

**Failed Node Impact:**
- Avg Degree Centrality: 0.5556
- Avg Betweenness Centrality: 0.5000
- Failure Rate: 0.2000

### cascade_gat

**Performance Metrics:**
- Final Accuracy: 0.6933
- Average Accuracy: 0.6328
- Accuracy Std Dev: 0.0345
- Robustness Score: 0.3220

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 139

**Error Propagation:**
- Mean Error Depth: 1.6667
- Max Error Depth: 4.0000
- Error Rate: 0.3333

**Failed Node Impact:**
- Avg Degree Centrality: 0.1944
- Avg Betweenness Centrality: 0.1736
- Failure Rate: 0.4000

### feedback_rewired_gat

**Performance Metrics:**
- Final Accuracy: 0.5767
- Average Accuracy: 0.6248
- Accuracy Std Dev: 0.0287
- Robustness Score: 0.3414

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 15
- Network Density: 0.1667

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 173

**Error Propagation:**
- Mean Error Depth: 2.6913
- Max Error Depth: 4.1000
- Error Rate: 0.4256

**Failed Node Impact:**
- Avg Degree Centrality: 0.1944
- Avg Betweenness Centrality: 0.1285
- Failure Rate: 0.4000

### ring_gat

**Performance Metrics:**
- Final Accuracy: 0.6933
- Average Accuracy: 0.6893
- Accuracy Std Dev: 0.0280
- Robustness Score: 0.4057

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 376

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.3014

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.2222
- Failure Rate: 0.4000

### mesh_gat

**Performance Metrics:**
- Final Accuracy: 0.7132
- Average Accuracy: 0.6898
- Accuracy Std Dev: 0.0276
- Robustness Score: 0.7306

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 1588

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.3038

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.5000

### small_world_gat

**Performance Metrics:**
- Final Accuracy: 0.7254
- Average Accuracy: 0.7187
- Accuracy Std Dev: 0.0380
- Robustness Score: 0.5258

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 688

**Error Propagation:**
- Mean Error Depth: 1.4333
- Max Error Depth: 1.7000
- Error Rate: 0.2681

**Failed Node Impact:**
- Avg Degree Centrality: 0.5556
- Avg Betweenness Centrality: 0.0833
- Failure Rate: 0.1000

### scale_free_gat

**Performance Metrics:**
- Final Accuracy: 0.6851
- Average Accuracy: 0.6753
- Accuracy Std Dev: 0.0436
- Robustness Score: 0.5185

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 9
- Total Messages Exchanged: 584

**Error Propagation:**
- Mean Error Depth: 1.4667
- Max Error Depth: 1.7000
- Error Rate: 0.3333

**Failed Node Impact:**
- Avg Degree Centrality: 0.5556
- Avg Betweenness Centrality: 0.0995
- Failure Rate: 0.4000

### tree_gat

**Performance Metrics:**
- Final Accuracy: 0.6528
- Average Accuracy: 0.6591
- Accuracy Std Dev: 0.0266
- Robustness Score: 0.4273

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 340

**Error Propagation:**
- Mean Error Depth: 2.1500
- Max Error Depth: 2.5000
- Error Rate: 0.3358

**Failed Node Impact:**
- Avg Degree Centrality: 0.1111
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.2000

### grid_gat

**Performance Metrics:**
- Final Accuracy: 0.7593
- Average Accuracy: 0.7360
- Accuracy Std Dev: 0.0405
- Robustness Score: 0.4422

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 362

**Error Propagation:**
- Mean Error Depth: 1.9667
- Max Error Depth: 2.5000
- Error Rate: 0.2448

**Failed Node Impact:**
- Avg Degree Centrality: 0.2500
- Avg Betweenness Centrality: 0.1010
- Failure Rate: 0.4000

### star_rf

**Performance Metrics:**
- Final Accuracy: 0.5193
- Average Accuracy: 0.5438
- Accuracy Std Dev: 0.0198
- Robustness Score: 0.4269

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 7
- Total Messages Exchanged: 268

**Error Propagation:**
- Mean Error Depth: 1.7000
- Max Error Depth: 1.7000
- Error Rate: 0.4723

**Failed Node Impact:**
- Avg Degree Centrality: 0.4074
- Avg Betweenness Centrality: 0.3333
- Failure Rate: 0.3000

### cascade_rf

**Performance Metrics:**
- Final Accuracy: 0.5438
- Average Accuracy: 0.5523
- Accuracy Std Dev: 0.0199
- Robustness Score: 0.3158

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 9
- Network Density: 0.1000

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 144

**Error Propagation:**
- Mean Error Depth: 2.0000
- Max Error Depth: 4.0000
- Error Rate: 0.4490

**Failed Node Impact:**
- Avg Degree Centrality: 0.1667
- Avg Betweenness Centrality: 0.1389
- Failure Rate: 0.2000

### feedback_rewired_rf

**Performance Metrics:**
- Final Accuracy: 0.5545
- Average Accuracy: 0.5469
- Accuracy Std Dev: 0.0243
- Robustness Score: 0.3672

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 14
- Network Density: 0.1556

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 200

**Error Propagation:**
- Mean Error Depth: 2.0417
- Max Error Depth: 3.3333
- Error Rate: 0.4502

**Failed Node Impact:**
- Avg Degree Centrality: 0.1667
- Avg Betweenness Centrality: 0.0556
- Failure Rate: 0.2000

### ring_rf

**Performance Metrics:**
- Final Accuracy: 0.5453
- Average Accuracy: 0.5400
- Accuracy Std Dev: 0.0197
- Robustness Score: 0.3775

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 20
- Network Density: 0.2222

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 312

**Error Propagation:**
- Mean Error Depth: 2.5000
- Max Error Depth: 2.5000
- Error Rate: 0.4416

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.2222
- Failure Rate: 0.4000

### mesh_rf

**Performance Metrics:**
- Final Accuracy: 0.4954
- Average Accuracy: 0.5509
- Accuracy Std Dev: 0.0167
- Robustness Score: 0.7038

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 90
- Network Density: 1.0000

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 1436

**Error Propagation:**
- Mean Error Depth: 0.9000
- Max Error Depth: 0.9000
- Error Rate: 0.5105

**Failed Node Impact:**
- Avg Degree Centrality: 1.0000
- Avg Betweenness Centrality: 0.0000
- Failure Rate: 0.6000

### small_world_rf

**Performance Metrics:**
- Final Accuracy: 0.5281
- Average Accuracy: 0.5465
- Accuracy Std Dev: 0.0172
- Robustness Score: 0.4992

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 40
- Network Density: 0.4444

**Reliability:**
- Max Failed Nodes: 3
- Total Messages Exchanged: 782

**Error Propagation:**
- Mean Error Depth: 1.4143
- Max Error Depth: 1.7000
- Error Rate: 0.4539

**Failed Node Impact:**
- Avg Degree Centrality: 0.4074
- Avg Betweenness Centrality: 0.0409
- Failure Rate: 0.3000

### scale_free_rf

**Performance Metrics:**
- Final Accuracy: 0.5649
- Average Accuracy: 0.5625
- Accuracy Std Dev: 0.0181
- Robustness Score: 0.5282

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 42
- Network Density: 0.4667

**Reliability:**
- Max Failed Nodes: 5
- Total Messages Exchanged: 620

**Error Propagation:**
- Mean Error Depth: 1.4429
- Max Error Depth: 1.7000
- Error Rate: 0.4465

**Failed Node Impact:**
- Avg Degree Centrality: 0.5556
- Avg Betweenness Centrality: 0.1119
- Failure Rate: 0.3000

### tree_rf

**Performance Metrics:**
- Final Accuracy: 0.5484
- Average Accuracy: 0.5554
- Accuracy Std Dev: 0.0154
- Robustness Score: 0.4105

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 18
- Network Density: 0.2000

**Reliability:**
- Max Failed Nodes: 4
- Total Messages Exchanged: 308

**Error Propagation:**
- Mean Error Depth: 2.3750
- Max Error Depth: 2.5000
- Error Rate: 0.4514

**Failed Node Impact:**
- Avg Degree Centrality: 0.3889
- Avg Betweenness Centrality: 0.6250
- Failure Rate: 0.2000

### grid_rf

**Performance Metrics:**
- Final Accuracy: 0.5316
- Average Accuracy: 0.5563
- Accuracy Std Dev: 0.0174
- Robustness Score: 0.4499

**Network Properties:**
- Number of Agents: 10
- Number of Edges: 26
- Network Density: 0.2889

**Reliability:**
- Max Failed Nodes: 6
- Total Messages Exchanged: 454

**Error Propagation:**
- Mean Error Depth: 1.9286
- Max Error Depth: 2.5000
- Error Rate: 0.4379

**Failed Node Impact:**
- Avg Degree Centrality: 0.2222
- Avg Betweenness Centrality: 0.0356
- Failure Rate: 0.3000

## Design Rules & Recommendations

Based on the experimental results, we derive the following design guidelines:

1. 🏆 **Best for Accuracy**: grid_gat achieved 0.759 accuracy. This topology is recommended when prediction quality is the primary concern.

2. 🛡️ **Best for Robustness**: mesh_gat achieved 0.731 robustness. Use this topology in environments with frequent failures or perturbations.

3. ⚖️ **Size Neutral**: Network size has minimal impact on accuracy (correlation: nan). Focus on topology design rather than scaling.

4. 💪 **Failure Resilience**: scale_free_neural maintains performance best under node failures. This topology has effective redundancy and graceful degradation.

5. ⚡ **Communication Efficient**: star_logistic, cascade_logistic, feedback_rewired_logistic, ring_logistic, tree_logistic, grid_logistic, star_gat, cascade_gat, feedback_rewired_gat, ring_gat, tree_gat, grid_gat, feedback_rewired_rf, ring_rf, tree_rf achieve above-average accuracy with below-average network density. These topologies minimize communication overhead.

## Conclusion

The **grid_gat** topology achieved the highest final accuracy of 0.7593. 
However, the optimal choice depends on the specific application requirements:

- For maximum accuracy: Choose the topology with highest final accuracy
- For reliability: Choose the topology with highest robustness score
- For efficiency: Choose topologies with low network density but high accuracy
- For scalability: Consider topologies that maintain performance as network size increases

---

*Report generated by Multi-Agent AI Systems Benchmarking Framework*