const routes = [
  { id: "large-model", label: "大模型" },
  { id: "meta-learning", label: "元学习" },
  { id: "meta-knowledge", label: "元知识" },
];

const datasets = [
  { id: "all", label: "全部数据集" },
  { id: "cifar10", label: "CIFAR-10" },
  { id: "cifar100", label: "CIFAR-100" },
  { id: "miniimagenet", label: "MiniImageNet" },
  { id: "isic", label: "ISIC / DermNet" },
  { id: "imagenet", label: "ImageNet" },
  { id: "cora", label: "Cora" },
  { id: "pubmed", label: "PubMed" },
  { id: "acm", label: "ACM" },
  { id: "imdb", label: "IMDB" },
  { id: "hyperspectral", label: "Hyperspectral Image" },
  { id: "missing-data", label: "Missing Data" },
  { id: "pde", label: "Function / PDE" },
  { id: "multiview", label: "Multi-View Data" },
];

const models = [
  {
    id: "resnet18",
    label: "ResNet-18",
    family: "Image Backbone",
    datasets: ["all", "cifar10", "cifar100", "imagenet"],
  },
  {
    id: "resnet50",
    label: "ResNet-50",
    family: "Image Backbone",
    datasets: ["all", "imagenet", "cifar10", "cifar100"],
  },
  {
    id: "vit-b16",
    label: "ViT-B/16",
    family: "Vision Transformer",
    datasets: ["all", "imagenet", "cifar10", "cifar100"],
  },
  {
    id: "mobilenetv2",
    label: "MobileNetV2",
    family: "Lightweight CNN",
    datasets: ["all", "cifar10", "cifar100", "imagenet"],
  },
  {
    id: "conv4",
    label: "Conv-4 Few-Shot Encoder",
    family: "Few-Shot Backbone",
    datasets: ["all", "miniimagenet", "isic"],
  },
  {
    id: "resnet12",
    label: "ResNet-12 Few-Shot Encoder",
    family: "Few-Shot Backbone",
    datasets: ["all", "miniimagenet", "isic"],
  },
  {
    id: "gcn",
    label: "GCN",
    family: "Graph Neural Network",
    datasets: ["all", "cora", "pubmed", "acm"],
  },
  {
    id: "gat",
    label: "GAT",
    family: "Graph Neural Network",
    datasets: ["all", "cora", "pubmed", "acm"],
  },
  {
    id: "multiview-encoder",
    label: "Multi-View Contrastive Encoder",
    family: "Multi-View Encoder",
    datasets: ["all", "acm", "imdb", "multiview"],
  },
  {
    id: "hsi-unet",
    label: "HSI Denoising U-Net",
    family: "Restoration Network",
    datasets: ["all", "hyperspectral"],
  },
  {
    id: "gain-generator",
    label: "GAIN Generator",
    family: "Imputation Model",
    datasets: ["all", "missing-data"],
  },
  {
    id: "hyperkan",
    label: "HyperKAN",
    family: "KAN / Scientific ML",
    datasets: ["all", "pde", "cifar10"],
  },
];

const levels = [
  {
    id: "data-sample",
    label: "数据样本层面",
    categories: [
      { id: "sample-selection", label: "样本自选择" },
      { id: "label-correction", label: "标签自校正" },
    ],
  },
  {
    id: "learning-process",
    label: "学习过程层面",
    categories: [
      { id: "model-construction", label: "模型自构建" },
      { id: "algorithm-design", label: "算法自设计" },
    ],
  },
  {
    id: "environment-task",
    label: "环境任务层面",
    categories: [
      { id: "environment-adaptation", label: "环境自适应" },
      { id: "task-transformation", label: "任务自转化" },
    ],
  },
];

const algorithms = [
  {
    id: "cmw-net",
    code: "CMW-Net",
    title: "Class-Aware Sample Weighting Mapping",
    level: "data-sample",
    category: "sample-selection",
    venue: "TPAMI 2023",
    year: 2023,
    routes: ["meta-learning"],
    datasets: ["cifar10", "cifar100", "imagenet"],
    description:
      "通过元模型学习类别感知的样本权重映射，用于缓解类别不平衡、标签噪声和真实偏置数据带来的鲁棒学习问题。",
    input: "带偏置或噪声的图像分类数据、基础分类网络、验证集或元数据。",
    output: "样本权重、鲁棒训练后的分类模型、准确率曲线。",
    tags: ["鲁棒学习", "样本重加权", "分类"],
  },
  {
    id: "ijcai26-sample",
    code: "IJCAI26",
    title: "Submitted Sample Selection Method",
    level: "data-sample",
    category: "sample-selection",
    venue: "IJCAI 2026 Submission",
    year: 2026,
    routes: ["meta-learning"],
    datasets: ["cifar10", "cifar100"],
    description:
      "预留给当前提交中的样本自选择算法，用于在平台中展示新增算法可插拔接入流程。",
    input: "训练数据、候选样本、验证反馈信号。",
    output: "样本选择策略、训练子集、评估报告。",
    tags: ["待公开", "样本选择"],
  },
  {
    id: "udw",
    code: "DWMNet",
    title: "Dual Weight-Net for Multi-View Graph Learning",
    level: "data-sample",
    category: "sample-selection",
    venue: "Information Fusion 2025",
    year: 2025,
    routes: ["meta-learning", "meta-knowledge"],
    datasets: ["acm", "imdb", "multiview"],
    description:
      "面向异构图和多视图数据，利用双权重网络自动调节视图与样本贡献，提升无监督表示学习质量。",
    input: "多视图图结构、节点特征、预训练配置或无监督训练配置。",
    output: "节点表示、视图权重、节点分类或聚类指标。",
    tags: ["多视图", "图学习", "无监督"],
  },
  {
    id: "nsmvc",
    code: "NS-MVC",
    title: "Mitigating Noisy Views for Multi-View Clustering",
    level: "data-sample",
    category: "sample-selection",
    venue: "CVPR 2024",
    year: 2024,
    routes: ["meta-knowledge"],
    datasets: ["multiview"],
    description:
      "针对实际多视图场景中的噪声视图副作用，自动识别并减弱低质量视图对自监督聚类的干扰。",
    input: "多视图特征、聚类配置、噪声视图设定。",
    output: "视图可靠性估计、聚类标签、NMI/ACC/ARI 指标。",
    tags: ["多视图", "聚类", "噪声视图"],
  },
  {
    id: "mlcgr",
    code: "MLC-GR",
    title: "Meta Label Correction with Generalization Regularizer",
    level: "data-sample",
    category: "label-correction",
    venue: "IJCAI 2025",
    year: 2025,
    routes: ["meta-learning"],
    datasets: ["cifar10"],
    description:
      "使用元网络进行噪声标签校正，并通过泛化正则约束提升在噪声标签场景下的稳定性。",
    input: "CIFAR-10 噪声标签数据、噪声率、噪声类型、训练超参数。",
    output: "校正标签、主网络分类结果、噪声鲁棒性指标。",
    tags: ["标签噪声", "元网络", "泛化正则"],
  },
  {
    id: "bmtd",
    code: "BMTD",
    title: "Bi-Level Multi-Teacher Distillation",
    level: "data-sample",
    category: "label-correction",
    venue: "AAAI 2025",
    year: 2025,
    routes: ["meta-learning", "meta-knowledge"],
    datasets: ["cora", "pubmed", "acm"],
    description:
      "通过双层优化融合多个教师模型，用于噪声节点分类中的标签校正和知识蒸馏。",
    input: "图数据、噪声节点标签、多教师编码器参数。",
    output: "校正后的节点标签、教师权重矩阵、节点分类结果。",
    tags: ["多教师", "图分类", "双层优化"],
  },
  {
    id: "narl",
    code: "NARL",
    title: "Noise-Aware Robust Loss Adjustment",
    level: "learning-process",
    category: "model-construction",
    venue: "TNNLS 2025",
    year: 2025,
    routes: ["meta-learning"],
    datasets: ["cifar10", "cifar100"],
    description:
      "通过噪声感知机制自动调整鲁棒损失，提高深度网络在不同噪声分布下的容忍能力。",
    input: "噪声分类数据、基础损失函数、噪声估计信号。",
    output: "自适应鲁棒损失、训练模型、噪声容忍评估。",
    tags: ["鲁棒损失", "噪声感知"],
  },
  {
    id: "hwnet",
    code: "HWNet",
    title: "Unknown Noise Adaptation for HSI Denoising",
    level: "learning-process",
    category: "model-construction",
    venue: "arXiv 2023",
    year: 2023,
    routes: ["meta-learning"],
    datasets: ["hyperspectral"],
    description:
      "面向高光谱图像去噪，学习适配未知噪声的网络结构与权重策略。",
    input: "高光谱图像、噪声退化数据、去噪训练配置。",
    output: "去噪图像、PSNR/SSIM 指标、噪声适配模型。",
    tags: ["高光谱", "去噪", "未知噪声"],
  },
  {
    id: "metakan",
    code: "MetaKAN",
    title: "Memory-Efficient KAN Training via Meta Learning",
    level: "learning-process",
    category: "model-construction",
    venue: "ICML 2025",
    year: 2025,
    routes: ["meta-learning"],
    datasets: ["pde", "cifar10"],
    description:
      "用元学习动态生成 KAN 样条函数参数，在函数拟合、图像分类和 PDE 求解中降低训练显存开销。",
    input: "函数拟合任务、图像分类任务或 PDE 任务配置。",
    output: "低显存 KAN 模型、任务性能、显存节省比例。",
    tags: ["KAN", "显存优化", "科学计算"],
  },
  {
    id: "metagain",
    code: "Meta-GAIN",
    title: "Meta-GAIN for Missing Data Imputation",
    level: "learning-process",
    category: "model-construction",
    venue: "AAAI 2026",
    year: 2026,
    routes: ["meta-learning"],
    datasets: ["missing-data"],
    description:
      "面向缺失数据填补，利用元学习提升生成式插补模型在不同缺失模式下的适应能力。",
    input: "缺失数据表、缺失掩码、插补训练配置。",
    output: "补全数据、插补误差、下游任务性能。",
    tags: ["缺失数据", "生成式插补"],
  },
  {
    id: "swcl",
    code: "SWCL",
    title: "Self-Weighted Contrastive Learning",
    level: "learning-process",
    category: "model-construction",
    venue: "NeurIPS 2023",
    year: 2023,
    routes: ["meta-knowledge"],
    datasets: ["multiview"],
    description:
      "在多视图对比学习中自动学习视图权重，缓解表示退化并提升聚类或分类效果。",
    input: "多视图样本特征、对比学习配置。",
    output: "多视图表示、视图权重、聚类指标。",
    tags: ["对比学习", "多视图", "元知识"],
  },
  {
    id: "gglg",
    code: "GGLG",
    title: "Global-Graph Guided Local-Graph Contrastive Learning",
    level: "learning-process",
    category: "model-construction",
    venue: "CVPR 2026",
    year: 2026,
    routes: ["meta-knowledge"],
    datasets: ["multiview"],
    description:
      "结合全局图指导与局部图加权机制，统一处理不完整和噪声多视图数据的聚类问题。",
    input: "不完整多视图数据、局部图和全局图配置。",
    output: "统一聚类表示、视图/图权重、聚类指标。",
    tags: ["图引导", "多视图", "聚类"],
  },
  {
    id: "anwl",
    code: "ANWL",
    title: "Adaptive Node-Level Weighted Learning",
    level: "learning-process",
    category: "model-construction",
    venue: "Neural Networks 2025",
    year: 2025,
    routes: ["meta-knowledge"],
    datasets: ["cora", "pubmed"],
    description:
      "针对有向图神经网络，自动学习节点级权重以提升有向图表示学习效果。",
    input: "有向图结构、节点特征、训练标签。",
    output: "节点级权重、有向图表示、节点分类结果。",
    tags: ["有向图", "节点权重"],
  },
  {
    id: "ndrl",
    code: "NDRL",
    title: "Node Diversity for Directed Graph Representation",
    level: "learning-process",
    category: "model-construction",
    venue: "IJCAI 2024",
    year: 2024,
    routes: ["meta-knowledge"],
    datasets: ["cora", "pubmed"],
    description:
      "探索节点多样性在有向图表示学习中的作用，并通过自动化权重设计提升节点分类。",
    input: "有向图、节点多样性度量、训练配置。",
    output: "多样性感知表示、分类结果、消融分析指标。",
    tags: ["节点多样性", "有向图"],
  },
  {
    id: "mlrsnet",
    code: "MLR-SNet",
    title: "Learned LR Schedules that Scale and Generalize",
    level: "learning-process",
    category: "algorithm-design",
    venue: "TPAMI 2023",
    year: 2023,
    routes: ["meta-learning"],
    datasets: ["cifar10", "imagenet"],
    description:
      "将学习率调度参数化为可学习映射，使其能随训练动态自动调整，并迁移到异构任务。",
    input: "网络结构、训练数据集、元训练任务和查询任务配置。",
    output: "可迁移学习率调度器、训练曲线、分类性能。",
    tags: ["学习率调度", "算法设计", "迁移"],
  },
  {
    id: "decap",
    code: "DeCap",
    title: "Prompt Learning for Few-Shot via Stable Diffusion",
    level: "environment-task",
    category: "task-transformation",
    venue: "TMLR 2025",
    year: 2025,
    routes: ["large-model", "meta-learning"],
    datasets: ["cifar10", "imagenet"],
    description:
      "通过 Stable Diffusion 生成多样且分类感知的提示与合成样本，增强少样本分类任务。",
    input: "少样本图像数据、类别名称、验证集标注、prompt pool。",
    output: "优化后的 prompt、合成训练集、少样本分类模型。",
    tags: ["Stable Diffusion", "少样本", "Prompt"],
  },
  {
    id: "tacm",
    code: "TACM",
    title: "Task Augmentation via Channel Mixture",
    level: "environment-task",
    category: "task-transformation",
    venue: "Neural Networks 2025",
    year: 2025,
    routes: ["meta-learning"],
    datasets: ["miniimagenet", "isic"],
    description:
      "通过通道混合进行任务增强，缓解 few-task meta-learning 中任务数量不足的问题。",
    input: "Few-shot 任务集合、support/query 划分、MAML 类训练配置。",
    output: "增强任务、元学习模型、few-shot 分类准确率。",
    tags: ["任务增强", "Few-shot", "MAML"],
  },
  {
    id: "mgrl",
    code: "MGRL",
    title: "Multiplex Graph Representation Learning",
    level: "learning-process",
    category: "model-construction",
    venue: "IJCAI 2024",
    year: 2024,
    routes: ["meta-knowledge"],
    datasets: ["acm", "imdb", "multiview"],
    description:
      "面向多监督和多视图学习，使用双层优化学习多路图表示，可作为平台扩展算法展示。",
    input: "多路图结构、节点特征、多监督信号。",
    output: "多路图表示、节点分类或聚类性能。",
    tags: ["多监督", "多视图", "扩展"],
  },
];

const state = {
  dataset: "all",
  model: "resnet18",
  route: "meta-learning",
  activeLevel: "data-sample",
  selectedByLevel: {
    "data-sample": {
      category: "sample-selection",
      algorithmId: "cmw-net",
    },
    "learning-process": {
      category: "model-construction",
      algorithmId: "narl",
    },
    "environment-task": {
      category: "task-transformation",
      algorithmId: "decap",
    },
  },
  training: {
    status: "idle",
    jobId: null,
    payload: null,
    stageIndex: -1,
    progress: 0,
    logs: [],
    metrics: {
      epoch: "--",
      loss: "--",
      accuracy: "--",
      elapsed: "00:00",
      device: "Jittor GPU",
    },
    artifacts: [],
    unsubscribe: null,
  },
};

const elements = {
  algorithmCount: document.querySelector("#algorithm-count"),
  datasetSelect: document.querySelector("#dataset-select"),
  modelSelect: document.querySelector("#model-select"),
  routeTabs: document.querySelector("#route-tabs"),
  automationMap: document.querySelector("#automation-map"),
  algorithmGrid: document.querySelector("#algorithm-grid"),
  gridHint: document.querySelector("#grid-hint"),
  currentLevel: document.querySelector("#current-level"),
  currentCategory: document.querySelector("#current-category"),
  currentModel: document.querySelector("#current-model"),
  currentCount: document.querySelector("#current-count"),
  resetButton: document.querySelector("#reset-button"),
  jobStatus: document.querySelector("#job-status"),
  jobDescription: document.querySelector("#job-description"),
  startTraining: document.querySelector("#start-training"),
  stopTraining: document.querySelector("#stop-training"),
  resetTraining: document.querySelector("#reset-training"),
  jobConfig: document.querySelector("#job-config"),
  trainingStages: document.querySelector("#training-stages"),
  trainingMetrics: document.querySelector("#training-metrics"),
  trainingLog: document.querySelector("#training-log"),
  jobId: document.querySelector("#job-id"),
  artifactList: document.querySelector("#artifact-list"),
};

const trainingStages = [
  "任务创建",
  "数据准备",
  "算法装配",
  "模型训练",
  "指标评估",
  "结果归档",
];

const trainingApi = {
  controllers: new Map(),

  createJob(payload) {
    const jobId = `AUTO6-${Date.now().toString(36).toUpperCase().slice(-7)}`;
    return {
      jobId,
      status: "running",
      stageIndex: 0,
      progress: 0,
      metrics: {
        epoch: 0,
        loss: "1.240",
        accuracy: "42.0%",
        elapsed: "00:00",
        device: "Jittor GPU",
      },
      artifacts: [],
      logs: [
        formatLog("INFO", `job ${jobId} created`),
        formatLog("INFO", `dataset=${payload.dataset.label}, route=${payload.route.label}`),
        formatLog("INFO", `model=${payload.model.label} (${payload.model.family})`),
        formatLog(
          "INFO",
          `pipeline=${payload.algorithms.map((item) => item.code).join(" -> ")}`,
        ),
      ],
    };
  },

  subscribeJob(jobId, payload, handlers) {
    let tick = 0;
    const totalTicks = trainingStages.length * 4;
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      tick += 1;
      const stageIndex = Math.min(
        trainingStages.length - 1,
        Math.floor((tick - 1) / 4),
      );
      const progress = Math.min(100, Math.round((tick / totalTicks) * 100));
      const stageProgress = tick % 4;
      const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
      const epoch = Math.min(120, tick * 5);
      const loss = Math.max(0.18, 1.24 - tick * 0.043).toFixed(3);
      const accuracy = Math.min(94.6, 42 + tick * 2.15).toFixed(1);
      const logs = [];

      if (stageProgress === 1) {
        logs.push(formatLog("STEP", `${trainingStages[stageIndex]} started`));
      }
      if (stageIndex === 2 && stageProgress === 2) {
        logs.push(
          formatLog(
            "INFO",
            `mounted ${payload.algorithms.length} algorithm module(s)`,
          ),
        );
      }
      if (stageIndex === 3) {
        logs.push(formatLog("TRAIN", `epoch=${epoch} loss=${loss} acc=${accuracy}%`));
      }
      if (stageIndex === 4 && stageProgress === 2) {
        logs.push(formatLog("EVAL", "validation metrics collected"));
      }

      handlers.update({
        status: "running",
        stageIndex,
        progress,
        logs,
        metrics: {
          epoch,
          loss,
          accuracy: `${accuracy}%`,
          elapsed: formatElapsed(elapsedSeconds),
          device: "Jittor GPU",
        },
        artifacts: [],
      });

      if (tick >= totalTicks) {
        window.clearInterval(timer);
        this.controllers.delete(jobId);
        handlers.complete({
          status: "completed",
          stageIndex: trainingStages.length - 1,
          progress: 100,
          logs: [
            formatLog("DONE", "training pipeline completed"),
            formatLog("DONE", "model package and report archived"),
          ],
          metrics: {
            epoch: 120,
            loss: "0.184",
            accuracy: "94.6%",
            elapsed: formatElapsed(Math.floor((Date.now() - startedAt) / 1000)),
            device: "Jittor GPU",
          },
          artifacts: [
            `模型权重: outputs/auto6ml/latest/${payload.model.id}.pkl`,
            "训练报告: outputs/auto6ml/latest/report.html",
            "指标面板: platform/runs/latest",
          ],
        });
      }
    }, 700);

    this.controllers.set(jobId, timer);
    return () => this.stopJob(jobId);
  },

  stopJob(jobId) {
    const timer = this.controllers.get(jobId);
    if (timer) {
      window.clearInterval(timer);
      this.controllers.delete(jobId);
    }
  },
};

function getCategory(id) {
  for (const level of levels) {
    const category = level.categories.find((item) => item.id === id);
    if (category) {
      return { ...category, level };
    }
  }
  return null;
}

function getLevel(id) {
  return levels.find((level) => level.id === id) || levels[0];
}

function getLevelState(levelId) {
  return state.selectedByLevel[levelId];
}

function matchesGlobalFilters(algorithm) {
  const routeMatch = algorithm.routes.includes(state.route);
  const datasetMatch =
    state.dataset === "all" || algorithm.datasets.includes(state.dataset);
  return routeMatch && datasetMatch;
}

function filteredAlgorithms(levelId, categoryId) {
  return algorithms.filter((algorithm) => {
    const levelMatch = algorithm.level === levelId;
    const categoryMatch = !categoryId || algorithm.category === categoryId;
    return levelMatch && categoryMatch && matchesGlobalFilters(algorithm);
  });
}

function categoryCount(categoryId) {
  return algorithms.filter((algorithm) => {
    return algorithm.category === categoryId && matchesGlobalFilters(algorithm);
  }).length;
}

function ensureLevelSelection(levelId) {
  const levelState = getLevelState(levelId);
  const selected = algorithms.find((algorithm) => algorithm.id === levelState.algorithmId);
  const selectedStillVisible =
    selected &&
    selected.level === levelId &&
    selected.category === levelState.category &&
    matchesGlobalFilters(selected);

  if (selectedStillVisible) return selected;
  return null;
}

function selectedAlgorithm(levelId = state.activeLevel) {
  return ensureLevelSelection(levelId);
}

function selectedAlgorithms() {
  return levels.map((level) => ({
    level,
    algorithm: selectedAlgorithm(level.id),
  }));
}

function compatibleModels() {
  return models.filter((model) => {
    return state.dataset === "all" || model.datasets.includes(state.dataset);
  });
}

function selectedModel() {
  const compatible = compatibleModels();
  const current = compatible.find((model) => model.id === state.model);
  if (current) return current;
  state.model = compatible[0]?.id || models[0].id;
  return models.find((model) => model.id === state.model) || models[0];
}

function selectedTrainingAlgorithms() {
  return selectedAlgorithms()
    .filter(({ algorithm }) => algorithm)
    .map(({ level, algorithm }) => {
      const category = getCategory(algorithm.category);
      return {
        levelId: level.id,
        levelLabel: level.label,
        categoryId: algorithm.category,
        categoryLabel: category.label,
        algorithmId: algorithm.id,
        code: algorithm.code,
        title: algorithm.title,
        venue: algorithm.venue,
      };
    });
}

function buildTrainingPayload() {
  const dataset = datasets.find((item) => item.id === state.dataset);
  const model = selectedModel();
  const route = routes.find((item) => item.id === state.route);
  const selected = selectedTrainingAlgorithms();
  return {
    dataset,
    model,
    route,
    algorithms: selected,
    pipelineName:
      selected.length > 0
        ? selected.map((item) => item.code).join(" + ")
        : "未配置训练算法",
    createdAt: new Date().toISOString(),
  };
}

function formatElapsed(totalSeconds) {
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function formatLog(level, message) {
  const time = new Date().toLocaleTimeString("zh-CN", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  return `[${time}] ${level.padEnd(5, " ")} ${message}`;
}

function appendTrainingLogs(logs) {
  state.training.logs = [...state.training.logs, ...logs].slice(-80);
}

function isTrainingRunning() {
  return state.training.status === "running";
}

function blockSelectionDuringTraining() {
  if (!isTrainingRunning()) return false;
  appendTrainingLogs([
    formatLog("WARN", "training is running; stop or reset the job before changing pipeline"),
  ]);
  renderTrainingPanel();
  return true;
}

function renderDatasets() {
  elements.datasetSelect.innerHTML = datasets
    .map((dataset) => `<option value="${dataset.id}">${dataset.label}</option>`)
    .join("");
  elements.datasetSelect.value = state.dataset;
}

function renderModels() {
  const compatible = compatibleModels();
  const model = selectedModel();
  elements.modelSelect.innerHTML = compatible
    .map(
      (item) => `
        <option value="${item.id}">${item.label} · ${item.family}</option>
      `,
    )
    .join("");
  elements.modelSelect.value = model.id;
}

function renderRoutes() {
  elements.routeTabs.innerHTML = routes
    .map(
      (route) => `
        <button class="route-button ${route.id === state.route ? "active" : ""}"
          type="button"
          data-route="${route.id}">
          ${route.label}
        </button>
      `,
    )
    .join("");
}

function renderAutomationMap() {
  elements.automationMap.innerHTML = levels
    .map(
      (level, index) => {
        const levelState = getLevelState(level.id);
        const visible = filteredAlgorithms(level.id, levelState.category);
        const selected = selectedAlgorithm(level.id);
        return `
        <article class="level-column ${level.id === state.activeLevel ? "active" : ""}">
          <header>
            <span>Level ${index + 1}</span>
            <h3>${level.label}</h3>
          </header>
          <div class="category-list">
            ${level.categories
              .map((category) => {
                const count = categoryCount(category.id);
                return `
                  <button class="category-button ${category.id === levelState.category ? "active" : ""}"
                    type="button"
                    data-level="${level.id}"
                    data-category="${category.id}">
                    <strong>${category.label}</strong>
                    <span>${count}</span>
                  </button>
                `;
              })
              .join("")}
          </div>
          <div class="level-algorithm-list">
            <div class="level-algorithm-list-title">当前层级算法</div>
            <button class="mini-algorithm-card none-card ${!selected ? "active" : ""}"
              type="button"
              data-level="${level.id}"
              data-none="true">
              <strong>不选择该层算法</strong>
              <span>该层级在当前方案中留空</span>
            </button>
            ${
              visible.length === 0
                ? '<div class="mini-empty">当前筛选暂无算法</div>'
                : visible
                    .map(
                      (algorithm) => `
                        <button class="mini-algorithm-card ${selected && selected.id === algorithm.id ? "active" : ""}"
                          type="button"
                          data-level="${level.id}"
                          data-algorithm="${algorithm.id}">
                          <strong>${algorithm.code}</strong>
                          <span>${algorithm.venue}</span>
                        </button>
                      `,
                    )
                    .join("")
            }
          </div>
        </article>
      `;
      },
    )
    .join("");
}

function renderAlgorithmGrid() {
  const selected = selectedAlgorithms();
  elements.gridHint.textContent = "三个层级的选择互相独立，每层可选一个算法，也可以留空。";

  elements.algorithmGrid.innerHTML = selected
    .map(
      ({ level, algorithm }) => {
        const levelState = getLevelState(level.id);
        const category = getCategory(levelState.category);
        if (!algorithm) {
          return `
            <button class="algorithm-card summary-card ${level.id === state.activeLevel ? "active" : ""}"
              type="button"
              data-level="${level.id}">
              <div class="card-topline">
                <span class="card-code">${level.label}</span>
                <span class="card-year">${category.label}</span>
              </div>
              <h3>未选择</h3>
              <p>该层级当前不接入算法，平台方案会保留这个层级的空位，不影响其他层级选择。</p>
              <div class="chip-row"><span class="chip">Optional</span></div>
            </button>
          `;
        }

        return `
        <button class="algorithm-card summary-card ${level.id === state.activeLevel ? "active" : ""}"
          type="button"
          data-level="${level.id}"
          data-algorithm="${algorithm.id}">
          <div class="card-topline">
            <span class="card-code">${level.label}</span>
            <span class="card-year">${category.label}</span>
          </div>
          <h3>${algorithm.code}</h3>
          <p>${algorithm.description}</p>
          <div class="chip-row">
            <span class="chip">${algorithm.venue}</span>
            ${algorithm.tags.slice(0, 2).map((tag) => `<span class="chip">${tag}</span>`).join("")}
          </div>
        </button>
      `;
      },
    )
    .join("");
}

function renderTrainingPanel() {
  const payload = state.training.payload || buildTrainingPayload();
  const selectedCount = payload.algorithms.length;
  const canStart = selectedCount > 0 && !isTrainingRunning();
  const statusLabel = {
    idle: "Idle",
    blocked: "Blocked",
    running: "Running",
    completed: "Completed",
    stopped: "Stopped",
  }[state.training.status];

  elements.jobStatus.textContent = statusLabel;
  elements.jobStatus.dataset.status = state.training.status;
  elements.jobDescription.textContent =
    selectedCount > 0
      ? `使用 ${payload.model.label} 作为模型骨干，将 ${selectedCount} 个层级算法装配为组合训练 pipeline：${payload.pipelineName}。`
      : `已选择模型 ${payload.model.label}，请再选择至少一个层级算法后启动训练。`;
  elements.startTraining.disabled = !canStart;
  elements.stopTraining.disabled = !isTrainingRunning();
  elements.resetTraining.disabled = state.training.status === "idle";
  elements.jobId.textContent = state.training.jobId || "No job";

  elements.jobConfig.innerHTML = [
    ["数据集", payload.dataset.label],
    ["模型", payload.model.label],
    ["技术路线", payload.route.label],
    ["Pipeline", payload.pipelineName],
    ["已选算法", `${selectedCount}/3`],
  ]
    .map(
      ([label, value]) => `
        <div>
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `,
    )
    .join("");

  renderTrainingStages();
  renderTrainingMetrics();
  renderTrainingLogs();
  renderArtifacts();
}

function renderTrainingStages() {
  elements.trainingStages.innerHTML = trainingStages
    .map((stage, index) => {
      let status = "pending";
      if (state.training.status === "completed") status = "done";
      else if (index < state.training.stageIndex) status = "done";
      else if (index === state.training.stageIndex && isTrainingRunning()) status = "active";
      else if (index === state.training.stageIndex && state.training.status === "stopped") {
        status = "stopped";
      }
      return `
        <div class="stage-item ${status}">
          <span>${index + 1}</span>
          <strong>${stage}</strong>
        </div>
      `;
    })
    .join("");
}

function renderTrainingMetrics() {
  const metrics = state.training.metrics;
  elements.trainingMetrics.innerHTML = [
    ["Epoch", metrics.epoch],
    ["Loss", metrics.loss],
    ["Accuracy", metrics.accuracy],
    ["Elapsed", metrics.elapsed],
    ["Device", metrics.device],
  ]
    .map(
      ([label, value]) => `
        <div>
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `,
    )
    .join("");
}

function renderTrainingLogs() {
  if (state.training.logs.length === 0) {
    elements.trainingLog.innerHTML =
      '<div class="log-line muted">等待提交训练任务...</div>';
    return;
  }

  elements.trainingLog.innerHTML = state.training.logs
    .map((line) => `<div class="log-line">${line}</div>`)
    .join("");
  elements.trainingLog.scrollTop = elements.trainingLog.scrollHeight;
}

function renderArtifacts() {
  const artifacts =
    state.training.artifacts.length > 0
      ? state.training.artifacts
      : [
          "训练任务提交后生成模型权重包",
          "训练任务提交后生成 HTML 训练报告",
          "训练任务提交后同步到平台结果面板",
        ];

  elements.artifactList.innerHTML = artifacts
    .map(
      (artifact) => `
        <div class="artifact-item ${state.training.artifacts.length > 0 ? "ready" : ""}">
          <span></span>
          <p>${artifact}</p>
        </div>
      `,
    )
    .join("");
}

function resetTrainingState() {
  if (state.training.unsubscribe) {
    state.training.unsubscribe();
  }
  state.training = {
    status: "idle",
    jobId: null,
    payload: null,
    stageIndex: -1,
    progress: 0,
    logs: [],
    metrics: {
      epoch: "--",
      loss: "--",
      accuracy: "--",
      elapsed: "00:00",
      device: "Jittor GPU",
    },
    artifacts: [],
    unsubscribe: null,
  };
}

function refreshTrainingPreview() {
  if (isTrainingRunning()) return;
  resetTrainingState();
}

function startTraining() {
  const payload = buildTrainingPayload();
  if (payload.algorithms.length === 0) {
    state.training.status = "blocked";
    state.training.payload = payload;
    appendTrainingLogs([formatLog("WARN", "select at least one algorithm before starting")]);
    renderTrainingPanel();
    return;
  }

  resetTrainingState();
  const job = trainingApi.createJob(payload);
  state.training = {
    ...state.training,
    status: job.status,
    jobId: job.jobId,
    payload,
    stageIndex: job.stageIndex,
    progress: job.progress,
    logs: job.logs,
    metrics: job.metrics,
    artifacts: job.artifacts,
  };
  state.training.unsubscribe = trainingApi.subscribeJob(job.jobId, payload, {
    update(snapshot) {
      state.training.status = snapshot.status;
      state.training.stageIndex = snapshot.stageIndex;
      state.training.progress = snapshot.progress;
      state.training.metrics = snapshot.metrics;
      appendTrainingLogs(snapshot.logs);
      renderTrainingPanel();
    },
    complete(snapshot) {
      state.training.status = snapshot.status;
      state.training.stageIndex = snapshot.stageIndex;
      state.training.progress = snapshot.progress;
      state.training.metrics = snapshot.metrics;
      state.training.artifacts = snapshot.artifacts;
      state.training.unsubscribe = null;
      appendTrainingLogs(snapshot.logs);
      renderTrainingPanel();
    },
  });
  renderTrainingPanel();
}

function stopTraining() {
  if (!isTrainingRunning()) return;
  if (state.training.unsubscribe) {
    state.training.unsubscribe();
  }
  state.training.status = "stopped";
  state.training.unsubscribe = null;
  appendTrainingLogs([formatLog("STOP", "job stopped by platform operator")]);
  renderTrainingPanel();
}

function renderStatus() {
  const level = getLevel(state.activeLevel);
  const levelState = getLevelState(state.activeLevel);
  const category = getCategory(levelState.category);
  const visible = filteredAlgorithms(state.activeLevel, levelState.category);
  const algorithm = selectedAlgorithm(state.activeLevel);
  const model = selectedModel();
  elements.currentLevel.textContent = level.label;
  elements.currentCategory.textContent = category.label;
  elements.currentModel.textContent = model.label;
  elements.currentCount.textContent = algorithm
    ? `${algorithm.code} / ${visible.length}`
    : `未选择 / ${visible.length}`;
  elements.algorithmCount.textContent = `${algorithms.length} algorithms`;
}

function renderAll() {
  renderRoutes();
  renderModels();
  renderAutomationMap();
  renderAlgorithmGrid();
  renderTrainingPanel();
  renderStatus();
}

function bindEvents() {
  elements.datasetSelect.addEventListener("change", (event) => {
    if (blockSelectionDuringTraining()) {
      elements.datasetSelect.value = state.dataset;
      return;
    }
    state.dataset = event.target.value;
    selectedModel();
    refreshTrainingPreview();
    renderAll();
  });

  elements.modelSelect.addEventListener("change", (event) => {
    if (blockSelectionDuringTraining()) {
      elements.modelSelect.value = state.model;
      return;
    }
    state.model = event.target.value;
    refreshTrainingPreview();
    renderAll();
  });

  elements.routeTabs.addEventListener("click", (event) => {
    const button = event.target.closest("[data-route]");
    if (!button) return;
    if (blockSelectionDuringTraining()) return;
    state.route = button.dataset.route;
    refreshTrainingPreview();
    renderAll();
  });

  elements.automationMap.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category]");
    const algorithmButton = event.target.closest("[data-algorithm]");
    const noneButton = event.target.closest("[data-none]");

    if (noneButton) {
      if (blockSelectionDuringTraining()) return;
      const levelId = noneButton.dataset.level;
      state.activeLevel = levelId;
      state.selectedByLevel[levelId].algorithmId = null;
      refreshTrainingPreview();
      renderAll();
      return;
    }

    if (algorithmButton) {
      if (blockSelectionDuringTraining()) return;
      const levelId = algorithmButton.dataset.level;
      const algorithm = algorithms.find((item) => item.id === algorithmButton.dataset.algorithm);
      state.activeLevel = levelId;
      state.selectedByLevel[levelId].algorithmId = algorithm.id;
      state.selectedByLevel[levelId].category = algorithm.category;
      refreshTrainingPreview();
      renderAll();
      return;
    }

    if (!button) return;
    if (blockSelectionDuringTraining()) return;
    const levelId = button.dataset.level;
    state.activeLevel = levelId;
    state.selectedByLevel[levelId].category = button.dataset.category;
    state.selectedByLevel[levelId].algorithmId = null;
    refreshTrainingPreview();
    renderAll();
  });

  elements.algorithmGrid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-level]");
    if (!button) return;
    if (blockSelectionDuringTraining()) return;
    const levelId = button.dataset.level;
    state.activeLevel = levelId;
    if (button.dataset.algorithm) {
      const algorithm = algorithms.find((item) => item.id === button.dataset.algorithm);
      state.selectedByLevel[levelId].algorithmId = algorithm.id;
      state.selectedByLevel[levelId].category = algorithm.category;
    }
    refreshTrainingPreview();
    renderAutomationMap();
    renderAlgorithmGrid();
    renderTrainingPanel();
    renderStatus();
  });

  elements.resetButton.addEventListener("click", () => {
    if (blockSelectionDuringTraining()) return;
    state.dataset = "all";
    state.model = "resnet18";
    state.route = "meta-learning";
    state.activeLevel = "data-sample";
    state.selectedByLevel = {
      "data-sample": {
        category: "sample-selection",
        algorithmId: "cmw-net",
      },
      "learning-process": {
        category: "model-construction",
        algorithmId: "narl",
      },
      "environment-task": {
        category: "task-transformation",
        algorithmId: "decap",
      },
    };
    elements.datasetSelect.value = state.dataset;
    elements.modelSelect.value = state.model;
    refreshTrainingPreview();
    renderAll();
  });

  elements.startTraining.addEventListener("click", startTraining);
  elements.stopTraining.addEventListener("click", stopTraining);
  elements.resetTraining.addEventListener("click", () => {
    resetTrainingState();
    renderTrainingPanel();
  });
}

renderDatasets();
renderModels();
renderAll();
bindEvents();
