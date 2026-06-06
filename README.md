# Auto-6ML

## Introduction

Auto^6ML is an open-source Jittor-based library for machine learning automation.
The project aims to integrate a series of auto-adaptive and meta-learning
algorithms into a unified platform, so that users can select datasets, choose an
automation scenario, and run the corresponding algorithm pipeline.

Our library is organized around the machine learning automation framework.
Under this framework, meta-learning methods are divided into three levels:

* Data and sample level: sample auto-selection and label auto-correction
* Learning process level: model auto-construction and algorithm auto-design
* Environment and task level: environment auto-adaptation and task
  auto-transformation

## Framework

### Machine Learning Automation

Machine learning automation aims to automatically improve data usage, label
quality, model structure, optimization strategy, environment adaptation, and
task construction.

### Meta-Learning

Meta-learning is the core technical route in Auto^6ML. The currently planned
methods are organized as follows.

#### Data and Sample Level

##### Sample Auto-Selection [Code]

* CMWNet - CMW-Net: Learning a Class-Aware Sample Weighting Mapping for Robust
  Deep Learning [TPAMI 2023](https://doi.org/10.1109/TPAMI.2023.3271451)
  [Code]
* IJCAI26 submission [submitted, paper URL unavailable] [Code]
* DWMNet - Unsupervised Multi-View Graph Representation Learning with Dual
  Weight-Net [Information Fusion 2025](https://www.sciencedirect.com/science/article/abs/pii/S1566253524004470)
  [Code]
* NS-MVC - Investigating and Mitigating the Side Effects of Noisy Views for
  auto-Supervised Clustering Algorithms in Practical Multi-View Scenarios
  [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Investigating_and_Mitigating_the_Side_Effects_of_Noisy_Views_for_CVPR_2024_paper.html)
  [Code]

##### Label Auto-Correction [Code]

* MLC-GR - Meta Label Correction with Generalization Regularizer
  [IJCAI 2025](https://www.ijcai.org/proceedings/2025/698) [Code]
* BMTD - Noisy Node Classification by Bi-level Optimization Based
  Multi-Teacher Distillation [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34095)
  [Code]

#### Learning Process Level

##### Model Auto-Construction [Code]

* NARL - Improve Noise Tolerance of Robust Loss via Noise-Awareness
  [TNNLS 2025](https://arxiv.org/abs/2301.07306) [Code]
* HWNet - Learning to Adapt Unknown Noise for Hyperspectral Image Denoising
  [arXiv 2023](https://arxiv.org/abs/2301.06081) [Code]
* Meta-KAN - Improving Memory Efficiency for Training KANs via Meta Learning
  [ICML 2025](https://proceedings.mlr.press/v267/zhao25s.html) [Code]
* Meta-GAIN - Meta-GAIN for Missing Data Imputation
  [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39798) [Code]
* SWCL - auto-Weighted Contrastive Learning among Multiple Views for Mitigating
  Representation Degeneration [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71375)
  [Code]
* GGLG - Global-Graph Guided and Local-Graph Weighted Contrastive Learning for
  Unified Clustering on Incomplete and Noisy Multi-View Data
  [CVPR 2026](https://arxiv.org/abs/2512.21516)
  [Code]
* ANWL - Adaptive Node-Level Weighted Learning for Directed Graph Neural
  Network [Neural Networks 2025](https://www.sciencedirect.com/science/article/pii/S0893608025002722)
  [Code]
* NDRL - Exploring the Role of Node Diversity in Directed Graph Representation
  Learning [IJCAI 2024](https://www.ijcai.org/proceedings/2024/229)
  [Code]

##### Algorithm Auto-Design [Code]

* Meta-LR-Schedule-Net - Learned LR Schedules that Scale and Generalize
  [TPAMI 2023](https://arxiv.org/abs/2007.14546) [Code]

#### Environment and Task Level

##### Environment Auto-Adaptation [Code]


##### Task Auto-Transformation [Code]

* DECAP - Diversity-Enhanced and Classification-Aware Prompt Learning for
  Few-Shot Learning via Stable Diffusion
  [TMLR 2025](https://openreview.net/forum?id=4CfliohyqK) [Code]
* TACM - Task Augmentation via Channel Mixture for Few-Task Meta-Learning
  [Neural Networks 2025](https://www.sciencedirect.com/science/article/pii/S0893608025004897)
  [Code]

## Related

### Multi-Supervision and Multi-View Learning

[1] Yudi Huang, Yujie Mo, Yujing Liu, Ci Nie, Guoqiu Wen, Xiaofeng Zhu.
Multiplex Graph Representation Learning via Bi-level Optimization.
[IJCAI 2024](https://www.ijcai.org/proceedings/2024/230).
