// ========== API Configuration ==========
const API_CONFIG = {
  baseUrl: '/api', // To be replaced
  endpoints: {
    // Dashboard
    getModelStatus: '/dashboard/status/',
    getBenchmarks: '/dashboard/benchmarks/',

    // Pipeline
    uploadTraining: '/pipeline/upload/',
    runClustering: '/pipeline/clustering/',
    runLabeling: '/pipeline/labeling/',
    runFeatures: '/pipeline/features/',
    runModelRace: '/pipeline/modelrace/',

    // Baseline Training
    trainFlamlBaseline: '/baseline/train_flaml/',
    trainTuneBaseline: '/baseline/train_tune/',
    trainAutofolioBaseline: '/baseline/train_autofolio/',
    trainRahaBaseline: '/baseline/train_raha/',
    getBaselineStatus: '/baseline/status/',
    getDataSplit: '/baseline/data_split/',

    // Recommend
    uploadInference: '/recommend/upload/',
    runRecommendFeatures: '/recommend/features/',
    getRecommendation: '/recommend/recommend/',
    compareBaselines: '/recommend/compare/',
    runImputation: '/recommend/impute/',
    runDownstream: '/recommend/downstream/',
    downloadResult: '/recommend/download/',

    // Evaluation Setup
    setupTestSet: '/recommend/setup_test_set/',
    setupUpload: '/recommend/setup_upload/',
    evaluationStatus: '/recommend/evaluation_status/',

    // Evaluation Metrics
    computeGroundTruthLabels: '/recommend/compute_ground_truth_labels/',
    evaluationMetrics: '/recommend/evaluation_metrics/',
    fullEvaluation: '/recommend/full_evaluation/',
  },

  // Set to true to use mock data for testing
  useMock: false,  // To be set to false
};

const PRIMARY_METHOD_KEY = 'ImputePilot';
const BRAND_NAME = 'ImputePilot';
const PRIMARY_METHOD_ALIASES = new Set([
  'ImputePilot',
  'ImputePilot',
  'ImputePilot',
  'ADART',
  'Adarts',
]);

function displayMethodName(method) {
  if (typeof method !== 'string') return method ?? '--';
  return PRIMARY_METHOD_ALIASES.has(method) ? BRAND_NAME : method;
}

function normalizePrimaryMethodKey(method) {
  if (typeof method !== 'string') return method;
  return PRIMARY_METHOD_ALIASES.has(method) ? PRIMARY_METHOD_KEY : method;
}

function normalizeBrandText(text) {
  if (typeof text !== 'string') return text;
  return text.replace(/\bImputePilot\b|\bImputePilot\b|\bADART\b|\bAdarts\b/g, BRAND_NAME);
}

// ========== State ==========
const AppState = {
  // Pipeline
  pipelineFiles: [],
  pipelineStep: 1,
  pipelineDatasetId: null,
  pipelineUploadResult: null,

  // Recommend
  recommendFiles: [],
  recommendStep: 1,
  recommendDatasetId: null,
  recommendUploadResult: null,
  recommendFeatureResults: null,
  recommendSeriesList: [],
  selectedRecoveredSeriesIndex: 0,
  lastImputationResults: null,
  selectedBaselines: ['FLAML', 'Tune', 'AutoFolio', 'RAHA'],  // Changed to array for multi-select support

  // Evaluation Mode (New)
  evaluationMode: 'upload',  // 'upload', 'test_set', 'complete_upload'
  missingRate: 0.20,
  missingPattern: 'random',
  groundTruthAvailable: false,
  groundTruthLabelsComputed: false,

  // Results cache
  clusterResults: null,
  labelingResults: null,
  externalDlStatus: null,
  featureResults: null,
  modelRaceResults: null,
  pipelineEvolution: null,  // Added: Pipeline Evolution data
  recommendResults: null,
  evaluationMetrics: null,  // New: Evaluation metrics results
  downstreamResults: {},  // New: Downstream evaluation results (cached by task)

  // Baseline status
  baselineStatus: {
    FLAML: { trained: false },
    Tune: { trained: false },
    AutoFolio: { trained: false },
    RAHA: { trained: false },
  },
  baselineTraining: {},
  dashboardBenchmark: {
    availableBenchmarks: [],
    benchmarkFile: '',
    generatedAt: '',
    methods: [],
    rows: [],
    datasetSearch: '',
    selectedDataset: '',
  },
};

// ========== Mock Data ==========
const MockData = {
  /**
   * Backend API: GET /dashboard/status
   * Expected Response Format:
   * {
   *   "lastTrained": "2025-12-25 14:30",
   *   "winningPipeline": "RandomForest + TSFRESH",
   *   "f1Score": 0.92
   * }
   */
  modelStatus: {
    lastTrained: '2025-12-25 14:30',
    winningPipeline: 'RandomForest + TSFRESH',
    f1Score: 0.92,
  },

  /**
   * Backend API: GET /dashboard/benchmarks
   * Expected Response Format:
   * {
   *   "available": true,
   *   "generated_at": "2026-03-01 10:30:12",
   *   "missing_rate": 0.1,
   *   "seed": 42,
   *   "methods": ["ImputePilot", "FLAML", "Tune", "AutoFolio", "RAHA"],
   *   "rows": [
   *     {
   *       "dataset": "Coffee",
   *       "method": "ImputePilot",
   *       "algo": "CDRec",
   *       "forecasting_rmse": 0.214,
   *       "classification_acc": 0.912,
   *       "status": "success"
   *     }
   *   ]
   * }
   */
  benchmarks: {
    available: true,
    generated_at: '2026-03-01 10:30:12',
    missing_rate: 0.1,
    seed: 42,
    methods: ['ImputePilot', 'FLAML', 'Tune', 'AutoFolio', 'RAHA'],
    rows: [
      {
        dataset: 'Coffee',
        method: 'ImputePilot',
        algo: 'CDRec',
        forecasting_rmse: 0.214,
        classification_acc: 0.912,
        status: 'success',
      },
      {
        dataset: 'Coffee',
        method: 'FLAML',
        algo: 'BRITS',
        forecasting_rmse: 0.268,
        classification_acc: 0.874,
        status: 'success',
      },
      {
        dataset: 'Coffee',
        method: 'Tune',
        algo: 'TRMF',
        forecasting_rmse: 0.246,
        classification_acc: 0.889,
        status: 'success',
      },
      {
        dataset: 'Coffee',
        method: 'AutoFolio',
        algo: 'SVDImp',
        forecasting_rmse: 0.301,
        classification_acc: 0.851,
        status: 'success',
      },
      {
        dataset: 'Coffee',
        method: 'RAHA',
        algo: 'TKCM',
        forecasting_rmse: 0.287,
        classification_acc: 0.862,
        status: 'success',
      },
      {
        dataset: 'Beef',
        method: 'ImputePilot',
        algo: 'CDRec',
        forecasting_rmse: 0.198,
        classification_acc: 0.927,
        status: 'success',
      },
      {
        dataset: 'Beef',
        method: 'FLAML',
        algo: 'TRMF',
        forecasting_rmse: 0.239,
        classification_acc: 0.889,
        status: 'success',
      },
      {
        dataset: 'Beef',
        method: 'Tune',
        algo: 'BRITS',
        forecasting_rmse: 0.228,
        classification_acc: 0.901,
        status: 'success',
      },
      {
        dataset: 'Beef',
        method: 'AutoFolio',
        algo: 'SVDImp',
        forecasting_rmse: 0.251,
        classification_acc: 0.882,
        status: 'success',
      },
      {
        dataset: 'Beef',
        method: 'RAHA',
        algo: 'TKCM',
        forecasting_rmse: 0.244,
        classification_acc: 0.887,
        status: 'success',
      },
    ],
  },

  /**
   * Backend API: POST /pipeline/clustering
   * Request Body:
   * {
   *   "datasetId": "dataset-123",
   *   "delta": 0.9,
   *   "rho": 0.20
   * }
   * Expected Response Format:
   * {
   *   "clusters": [
   *     { "id": 1, "name": "Cluster 1", "count": 256, "rho": 0.90, "shapePreview": [0.1, 0.3, ...] },
   *     { "id": 2, "name": "Cluster 2", "count": 128, "rho": 0.91, "shapePreview": [0.2, 0.1, ...] }
   *   ]
   * }
   */
  clusterResults: [
    { id: 1, name: 'Cluster 1', count: 256, rho: 0.90, shapePreview: [0.10, 0.16, 0.22, 0.35, 0.44, 0.40, 0.30, 0.18, 0.11] },
    { id: 2, name: 'Cluster 2', count: 128, rho: 0.91, shapePreview: [0.35, 0.28, 0.19, 0.12, 0.15, 0.24, 0.31, 0.39, 0.45] },
    { id: 3, name: 'Cluster 3', count: 64, rho: 0.92, shapePreview: [0.05, 0.08, 0.14, 0.20, 0.28, 0.36, 0.44, 0.52, 0.58] },
    { id: 4, name: 'Cluster 4', count: 32, rho: 0.90, shapePreview: [0.42, 0.41, 0.38, 0.31, 0.25, 0.22, 0.24, 0.30, 0.37] },
    { id: 5, name: 'Cluster 5', count: 16, rho: 0.90, shapePreview: [0.22, 0.25, 0.29, 0.33, 0.34, 0.30, 0.26, 0.21, 0.18] },
  ],

  /**
   * Backend API: POST /pipeline/labeling
   * Request Body:
   * {
   *   "datasetId": "dataset-123",
   *   "algorithms": ["STMVL", "CDRec", "SVDImp", "TRMF", "TKCM", ...]
   * }
   * Expected Response Format:
   * {
   *   "labelingResults": [
   *     { "id": 1, "name": "Cluster 1", "count": 256, "rmse": 0.01, "bestAlgo": "STMVL" },
   *     { "id": 2, "name": "Cluster 2", "count": 128, "rmse": 0.02, "bestAlgo": "TKCM" }
   *   ]
   * }
   */
  labelingResults: [
    { id: 1, name: 'Cluster 1', count: 256, rmse: 0.01, bestAlgo: 'STMVL' },
    { id: 2, name: 'Cluster 2', count: 128, rmse: 0.02, bestAlgo: 'TKCM' },
    { id: 3, name: 'Cluster 3', count: 64, rmse: 0.03, bestAlgo: 'MRNN' },
    { id: 4, name: 'Cluster 4', count: 32, rmse: 0.04, bestAlgo: 'IIM' },
    { id: 5, name: 'Cluster 5', count: 16, rmse: 0.05, bestAlgo: 'MPIN' },
  ],

  /**
    * Backend API: POST /pipeline/features
    * Request Body:
    * {
    *   "datasetId": "dataset-123"
    * }
    * Expected Response Format:
    * {
    *   "featureImportance": [
    *     { "name": "catch22", "value": 22 },
    *     { "name": "tsfresh", "value": 783 },
    *     { "name": "topological", "value": 3 }
    *   ],
    *   "featurePreview": {
    *     "catch22": {
    *       "dataset": "Coffee",
    *       "idColumn": "Time Series ID",
    *       "totalFeatures": 22,
    *       "sampleColumns": ["DN_HistogramMode_5", "CO_f1ecac"],
    *       "rows": [{ "Time Series ID": 0, "DN_HistogramMode_5": 0.1, "CO_f1ecac": 0.9 }]
    *     }
    *   }
    * }
    */
  featureResults: {
    featureImportance: [
      { name: 'catch22', value: 22, datasetsProcessed: 1 },
      { name: 'tsfresh', value: 783, datasetsProcessed: 1 },
      { name: 'topological', value: 3, datasetsProcessed: 1 },
    ],
    previewRows: 3,
    previewCols: 6,
    featurePreview: {
      catch22: {
        dataset: 'Coffee',
        idColumn: 'Time Series ID',
        totalFeatures: 22,
        sampleColumns: ['DN_HistogramMode_5', 'CO_f1ecac', 'SB_BinaryStats_mean_longstretch1', 'FC_LocalSimple_mean1_tauresrat'],
        truncated: true,
        rows: [
          { 'Time Series ID': 0, DN_HistogramMode_5: 0.238, CO_f1ecac: 0.912, SB_BinaryStats_mean_longstretch1: 12, FC_LocalSimple_mean1_tauresrat: 0.411 },
          { 'Time Series ID': 1, DN_HistogramMode_5: 0.214, CO_f1ecac: 0.877, SB_BinaryStats_mean_longstretch1: 9, FC_LocalSimple_mean1_tauresrat: 0.389 },
          { 'Time Series ID': 2, DN_HistogramMode_5: 0.195, CO_f1ecac: 0.844, SB_BinaryStats_mean_longstretch1: 10, FC_LocalSimple_mean1_tauresrat: 0.402 },
        ],
      },
      tsfresh: {
        dataset: 'Coffee',
        idColumn: 'Time Series ID',
        totalFeatures: 783,
        sampleColumns: ['Values__mean', 'Values__variance', 'autocorrelation__lag_1', 'fft_coefficient__attr_"abs"__coeff_0'],
        truncated: true,
        rows: [
          { 'Time Series ID': 0, Values__mean: 0.582, Values__variance: 0.041, autocorrelation__lag_1: 0.972, 'fft_coefficient__attr_"abs"__coeff_0': 17.44 },
          { 'Time Series ID': 1, Values__mean: 0.601, Values__variance: 0.038, autocorrelation__lag_1: 0.965, 'fft_coefficient__attr_"abs"__coeff_0': 18.03 },
          { 'Time Series ID': 2, Values__mean: 0.563, Values__variance: 0.045, autocorrelation__lag_1: 0.958, 'fft_coefficient__attr_"abs"__coeff_0': 16.91 },
        ],
      },
      topological: {
        dataset: 'Coffee',
        idColumn: 'Time Series ID',
        totalFeatures: 3,
        sampleColumns: ['topological_1', 'topological_2', 'topological_3'],
        truncated: false,
        rows: [
          { 'Time Series ID': 0, topological_1: 0.812, topological_2: 0.463, topological_3: 0.121 },
          { 'Time Series ID': 1, topological_1: 0.798, topological_2: 0.441, topological_3: 0.118 },
          { 'Time Series ID': 2, topological_1: 0.821, topological_2: 0.457, topological_3: 0.116 },
        ],
      },
    },
  },

  /**
   * Backend API: POST /pipeline/modelrace
   * Request Body:
   * {
   *   "datasetId": "dataset-123",
   *   "alpha": 0.50,
   *   "beta": 0.50,
   *   "gamma": 0.75,
   *   "seedPipelines": 100,
   *   "pValue": 0.50
   * }
   * Expected Response Format:
   * {
   *   "pipelineResults": [
   *     { "name": "RandomForest", "f1": 0.90, "rank": 1 },
   *     { "name": "CatBoost", "f1": 0.88, "rank": 2 }
   *   ],
   *   "evolution": [
   *     { "round": 1, "candidates": 64, "eliminated": 14, "bestF1": 0.72, "bestPipeline": "CatBoost + Canonical" },
   *     { "round": 2, "candidates": 50, "eliminated": 12, "bestF1": 0.78, "bestPipeline": "RandomForest + Trends" }
   *   ]
   * }
   */
  modelRaceResults: [
    { name: 'RandomForest', f1: 0.90, rank: 1 },
    { name: 'CatBoost', f1: 0.88, rank: 2 },
    { name: 'KNN', f1: 0.86, rank: 3 },
    { name: 'MLP', f1: 0.84, rank: 4 },
    { name: 'Decision Tree', f1: 0.82, rank: 5 },
  ],

  // Pipeline Evolution data (part of POST /pipeline/modelrace response)
  pipelineEvolution: [
    { round: 1, candidates: 64, eliminated: 14, bestF1: 0.72, bestPipeline: 'CatBoost + Canonical' },
    { round: 2, candidates: 50, eliminated: 12, bestF1: 0.78, bestPipeline: 'RandomForest + Trends' },
    { round: 3, candidates: 38, eliminated: 10, bestF1: 0.84, bestPipeline: 'RandomForest + Topological' },
    { round: 4, candidates: 28, eliminated: 8, bestF1: 0.88, bestPipeline: 'RandomForest + TSFRESH' },
    { round: 5, candidates: 20, eliminated: 15, bestF1: 0.90, bestPipeline: 'RandomForest + TSFRESH' },
  ],

  /**
   * Backend API: POST /recommend/recommend
   * Request Body:
   * {
   *   "datasetId": "dataset-123"
   * }
   * Expected Response Format:
   * {
   *   "ranking": [
   *     { "rank": 1, "algo": "ROSL" },
   *     { "rank": 2, "algo": "SVDImp" }
   *   ],
   *   "votingMatrix": [
   *     { "algo": "ROSL", "p1": 0.35, "p2": 0.40, "p3": 0.38, "avg": 0.38 },
   *     { "algo": "SVDImp", "p1": 0.25, "p2": 0.22, "p3": 0.28, "avg": 0.25 }
   *   ],
   *   "ImputePilot": { "algo": "ROSL", "f1": 0.95, "accuracy": 0.93 },
   *   "baselines": {
   *     "FLAML": { "algo": "CDRec", "f1": 0.82, "accuracy": 0.80 },
   *     "Tune": { "algo": "BRITS", "f1": 0.78, "accuracy": 0.76 },
   *     "AutoFolio": { "algo": "TRMF", "f1": 0.75, "accuracy": 0.73 },
   *     "RAHA": { "algo": "SVDImp", "f1": 0.80, "accuracy": 0.78 }
   *   }
   * }
   */
  recommendResults: {
    ranking: [
      { rank: 1, algo: 'ROSL' },
      { rank: 2, algo: 'SVDImp' },
      { rank: 3, algo: 'GROUSE' },
      { rank: 4, algo: 'SoftImp' },
      { rank: 5, algo: 'DynaMMo' },
    ],
    pipelinesConfigured: 5,
    pipelinesUsed: 4,
    pipelineHeaders: ['P1', 'P2', 'P3', 'P4'],
    votingMatrix: [
      { algo: 'ROSL', pipelineScores: [0.35, 0.40, 0.38, 0.39], avg: 0.38, p1: 0.35, p2: 0.40, p3: 0.38 },
      { algo: 'SVDImp', pipelineScores: [0.25, 0.22, 0.28, 0.24], avg: 0.25, p1: 0.25, p2: 0.22, p3: 0.28 },
      { algo: 'GROUSE', pipelineScores: [0.20, 0.18, 0.17, 0.19], avg: 0.19, p1: 0.20, p2: 0.18, p3: 0.17 },
      { algo: 'SoftImp', pipelineScores: [0.12, 0.12, 0.10, 0.11], avg: 0.11, p1: 0.12, p2: 0.12, p3: 0.10 },
      { algo: 'DynaMMo', pipelineScores: [0.08, 0.08, 0.07, 0.07], avg: 0.08, p1: 0.08, p2: 0.08, p3: 0.07 },
    ],
    ImputePilot: { algo: 'ROSL', confidence: 0.95, inference_time_ms: 12.5 },
  },

  // Baseline data (included in POST /recommend/recommend response as "baselines" field)
  // Used for mock mode only; in production, use response.baselines
  baselineData: {
    FLAML: { trained: true, algo: 'CDRec', confidence: 0.82, inference_time_ms: 8.3, f1_train: 0.85, best_estimator: 'lgbm' },
    Tune: { trained: true, algo: 'BRITS', confidence: 0.78, inference_time_ms: 10.5, f1_train: 0.80, best_estimator: 'RandomForest' },
    AutoFolio: { trained: true, algo: 'TRMF', confidence: 0.76, inference_time_ms: 11.2, f1_train: 0.79, best_estimator: 'extra_trees' },
    RAHA: { trained: true, algo: 'SVDImp', confidence: 0.74, inference_time_ms: 12.8, f1_train: 0.78, best_estimator: 'cosine_similarity' },
  },

  /**
   * Backend API: POST /recommend/impute
   * Request Body:
   * {
   *   "datasetId": "dataset-123",
   *   "algorithm": "ROSL"
   * }
   * Expected Response Format:
   * {
   *   "algo": "ROSL",
   *   "missingPoints": 300,
   *   "recoveryRate": "100%",
   *   "processingTime": "1.12s",
   *   "comparison": [
   *     { "method": "ImputePilot", "algo": "ROSL", "rmse": 0.0123, "mae": 0.0111, "runtime": "2.6s", "improvement": null },
   *     { "method": "FLAML", "algo": "CDRec", "rmse": 0.0234, "mae": 0.0222, "runtime": "3.2s", "improvement": "+43%" }
   *   ]
   * }
   */
  imputationResults: {
    algo: 'ROSL',
    missingPoints: 300,
    recoveryRate: '100%',
    processingTime: '1.12s',
    comparison: [
      { method: 'ImputePilot', algo: 'ROSL', rmse: 0.0123, mae: 0.0111, runtime: '2.6s', improvement: null },
      { method: 'FLAML', algo: 'CDRec', rmse: 0.0234, mae: 0.0222, runtime: '3.2s', improvement: '+43%' },
      { method: 'Tune', algo: 'BRITS', rmse: 0.0345, mae: 0.0333, runtime: '3.6s', improvement: '+54%' },
      { method: 'RAHA', algo: 'SVDImp', rmse: 0.0456, mae: 0.0444, runtime: '4.2s', improvement: '+60%' },
      { method: 'AutoFolio', algo: 'TRMF', rmse: 0.567, mae: 0.0555, runtime: '5.0s', improvement: '+70%' },
    ],
  },

  /**
   * Backend API: POST /recommend/downstream
   * Request Body:
   * {
   *   "datasetId": "dataset-123",
   *   "task": "forecasting" | "classification"
   * }
   * Expected Response Format (Updated for three-column comparison):
   * {
   *   "groundTruth": { "value": 0.32, "std": 0.05 },
   *   "withImputePilot": { "value": 0.45, "std": 0.08 },
   *   "withoutImputePilot": { "value": 0.97, "std": 0.15 },
   *   "improvement": 53.6,
   *   "gapToOptimal": 40.6,
   *   "n_evaluated": 50
   * }
   */
  downstreamResults: {
    forecasting: {
      groundTruth: { value: 0.32, std: 0.05 },
      withImputePilot: { value: 0.45, std: 0.08 },
      withoutImputePilot: { value: 0.97, std: 0.15 },
      improvement: 53.6,
      gapToOptimal: 40.6,
      n_evaluated: 50
    },
    classification: {
      groundTruth: { value: 0.95, std: 0.03 },
      withImputePilot: { value: 0.88, std: 0.06 },
      withoutImputePilot: { value: 0.72, std: 0.12 },
      improvement: 22.2,
      gapToOptimal: 7.4,
      n_evaluated: 50
    },
  },

  /**
   * Backend API: GET /recommend/evaluation_metrics/
   * Expected Response Format:
   * {
   *   "results": {
   *     "ImputePilot": {
   *       "metrics": {
   *         "accuracy": 0.82,
   *         "macro_f1": 0.78,
   *         "weighted_f1": 0.80,
   *         "mrr": 0.85,
   *         "top_k_accuracy": { "top_1": 0.82, "top_3": 0.94, "top_5": 0.98 }
   *       },
   *       "predictions": [...]
   *     },
   *     "FLAML": {
   *       "metrics": {
   *         "accuracy": 0.68,
   *         "macro_f1": 0.62,
   *         "weighted_f1": 0.65,
   *         "mrr": 0.72,
   *         "top_k_accuracy": { "top_1": 0.68, "top_3": 0.85, "top_5": 0.92 }
   *       },
   *       "predictions": [...]
   *     }
   *   },
   *   "ground_truth_labels": [...],
   *   "n_samples": 50
   * }
   */
  evaluationMetrics: {
    results: {
      'ImputePilot': {
        metrics: {
          accuracy: 0.82,
          macro_f1: 0.78,
          weighted_f1: 0.80,
          mrr: 0.85,
          top_k_accuracy: { top_1: 0.82, top_3: 0.94, top_5: 0.98 }
        }
      },
      'FLAML': {
        metrics: {
          accuracy: 0.68,
          macro_f1: 0.62,
          weighted_f1: 0.65,
          mrr: 0.72,
          top_k_accuracy: { top_1: 0.68, top_3: 0.85, top_5: 0.92 }
        }
      }
    },
    ground_truth_labels: ['ROSL', 'CDRec', 'BRITS', 'SVDImp', 'ROSL'],
    n_samples: 50
  },
};

/**
 * Backend API: POST /pipeline/upload
 * Request: multipart/form-data with field "files" containing one or more files
 * Expected Response Format:
 * {
 *   "datasetId": "dataset-123"
 * }
 * 
 * Backend API: POST /recommend/upload
 * Request: multipart/form-data with field "files" containing one file
 * Expected Response Format:
 * {
 *   "datasetId": "dataset-456"
 * }
 * 
 * Backend API: GET /recommend/download?datasetId=xxx
 * Response: File download (binary stream)
 */

// ========== API Helper Functions ==========
async function apiCall(endpoint, method = 'GET', body = null) {
  if (API_CONFIG.useMock) {
    console.log(`[Mock API] ${method} ${endpoint}`, body);
    await simulateDelay(1000 + Math.random() * 1000);
    return null; // Will use mock data
  }

  const options = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };

  if (body && method !== 'GET') {
    options.body = JSON.stringify(body);
  }

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}${endpoint}`, options);
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    throw error;
  }
}

async function uploadFiles(endpoint, files) {
  if (API_CONFIG.useMock) {
    console.log(`[Mock API] Upload to ${endpoint}`, files);
    await simulateDelay(1500);
    return { datasetId: 'mock-dataset-' + Date.now() };
  }

  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`Upload Error: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
}

// ========== Navigation ==========
document.querySelectorAll('.menu-item').forEach(item => {
  item.addEventListener('click', function () {
    document.querySelectorAll('.menu-item').forEach(m => m.classList.remove('active'));
    this.classList.add('active');

    const pageId = this.dataset.page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${pageId}`).classList.add('active');
    updatePipelineClusterLayoutState(pageId === 'pipeline' && [2, 3, 4, 5].includes(AppState.pipelineStep));
    updateRecommendFeatureLayoutState(pageId === 'recommend' && AppState.recommendStep === 2);

    // Load page-specific data
    if (pageId === 'dashboard') {
      loadDashboardData();
    }
  });
});

// ==================== Utilities ====================

// ========== Dashboard ==========
async function loadDashboardData() {
  try {
    // Load Model Status
    let statusData;
    if (API_CONFIG.useMock) {
      statusData = MockData.modelStatus;
    } else {
      statusData = await apiCall(API_CONFIG.endpoints.getModelStatus);
    }

    // Update Model Status DOM
    const lastTrained = document.getElementById('last-trained');
    const winningPipeline = document.getElementById('winning-pipeline');
    const f1Score = document.getElementById('f1-score');

    if (lastTrained) lastTrained.textContent = statusData.lastTrained || '--';
    if (winningPipeline) winningPipeline.textContent = statusData.winningPipeline || '--';
    if (f1Score) f1Score.textContent = statusData.f1Score ? statusData.f1Score.toFixed(2) : '--';

    // Load Benchmarks
    const benchmarkData = await fetchDashboardBenchmark();
    renderRealworldBenchmarks(benchmarkData, { preserveSelection: true });

  } catch (error) {
    console.error('Failed to load dashboard data:', error);
  }
}

async function fetchDashboardBenchmark(benchmarkFile = '') {
  if (API_CONFIG.useMock) {
    return MockData.benchmarks;
  }
  let endpoint = API_CONFIG.endpoints.getBenchmarks;
  if (benchmarkFile) {
    endpoint += `?benchmark_file=${encodeURIComponent(benchmarkFile)}`;
  }
  return apiCall(endpoint);
}

function getRealworldControlElements() {
  return {
    summaryEmpty: document.getElementById('rw-summary-empty'),
    summaryCharts: document.querySelector('#page-dashboard .summary-charts'),
    summaryMeta: document.getElementById('rw-summary-meta'),
    benchmarkSelect: document.getElementById('rw-benchmark-select'),
    datasetSearchInput: document.getElementById('rw-dataset-search'),
    datasetSelect: document.getElementById('rw-dataset-select'),
    forecastTitle: document.getElementById('rw-forecasting-title'),
    classificationTitle: document.getElementById('rw-classification-title'),
  };
}

function setSelectOptions(selectEl, options, selectedValue, emptyLabel = '--') {
  if (!selectEl) return;
  const previousValue = selectedValue ?? selectEl.value;
  selectEl.innerHTML = '';
  if (!Array.isArray(options) || !options.length) {
    const emptyOption = document.createElement('option');
    emptyOption.value = '';
    emptyOption.textContent = emptyLabel;
    selectEl.appendChild(emptyOption);
    selectEl.value = '';
    selectEl.disabled = true;
    return;
  }
  options.forEach(({ value, label }) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    selectEl.appendChild(option);
  });
  const allValues = new Set(options.map(item => item.value));
  const nextValue = allValues.has(previousValue) ? previousValue : options[0].value;
  selectEl.value = nextValue;
  selectEl.disabled = false;
}

function formatBenchmarkRunLabel(item) {
  if (!item) return 'Unknown run';
  const generatedAt = item.generated_at || item.generatedAt;
  if (generatedAt) return generatedAt;
  const mtime = item.mtime;
  if (mtime) {
    const date = new Date(Number(mtime) * 1000);
    if (!Number.isNaN(date.getTime())) return date.toISOString().slice(0, 19).replace('T', ' ');
  }
  return item.file || 'Unknown run';
}

function getAvailableDatasets(rows) {
  const names = Array.from(new Set((rows || []).map(r => r?.dataset).filter(Boolean)));
  names.sort((a, b) => a.localeCompare(b));
  return names;
}

function getMethodListFromBenchmark(benchmarkData) {
  const defaultMethods = ['ImputePilot', 'FLAML', 'Tune', 'AutoFolio', 'RAHA'];
  const providedMethods = Array.isArray(benchmarkData?.methods) && benchmarkData.methods.length
    ? benchmarkData.methods
    : defaultMethods;
  const normalizedProvided = providedMethods
    .map(method => normalizePrimaryMethodKey(method))
    .filter(Boolean);
  const methodSet = new Set(normalizedProvided);
  const ordered = defaultMethods
    .map(method => normalizePrimaryMethodKey(method))
    .filter(method => methodSet.has(method));
  normalizedProvided.forEach(method => {
    if (!ordered.includes(method)) ordered.push(method);
  });
  return ordered;
}

function showRealworldBenchmarkEmpty(message) {
  const { summaryEmpty, summaryCharts, summaryMeta, forecastTitle, classificationTitle } = getRealworldControlElements();
  if (summaryEmpty) {
    summaryEmpty.style.display = 'flex';
    summaryEmpty.textContent = message;
  }
  if (summaryCharts) summaryCharts.style.display = 'none';
  if (summaryMeta) summaryMeta.textContent = '';
  if (forecastTitle) forecastTitle.textContent = 'Forecasting RMSE by Method';
  if (classificationTitle) classificationTitle.textContent = 'Classification Accuracy by Method';
  destroyRealworldCharts();
}

function renderSelectedRealworldDataset() {
  const {
    summaryEmpty,
    summaryCharts,
    summaryMeta,
    datasetSearchInput,
    datasetSelect,
    forecastTitle,
    classificationTitle,
  } = getRealworldControlElements();
  const state = AppState.dashboardBenchmark;
  const rows = state.rows || [];
  const methods = state.methods || [];

  if (!rows.length || !methods.length) {
    showRealworldBenchmarkEmpty('Run the RealWorld benchmark to populate dataset-level charts.');
    return;
  }

  if (datasetSearchInput && datasetSearchInput.value !== state.datasetSearch) {
    datasetSearchInput.value = state.datasetSearch || '';
  }

  const allDatasets = getAvailableDatasets(rows);
  const query = (state.datasetSearch || '').trim().toLowerCase();
  const filteredDatasets = query
    ? allDatasets.filter(name => name.toLowerCase().includes(query))
    : allDatasets;

  const datasetOptions = filteredDatasets.map(name => ({ value: name, label: name }));
  setSelectOptions(datasetSelect, datasetOptions, state.selectedDataset, 'No matching dataset');
  if (datasetSelect) {
    state.selectedDataset = datasetSelect.value || '';
  }

  if (!state.selectedDataset) {
    showRealworldBenchmarkEmpty('No dataset matches the current search. Try another keyword.');
    return;
  }

  const grouped = {};
  rows.forEach(row => {
    if (!row || !row.dataset || !row.method) return;
    const method = normalizePrimaryMethodKey(row.method);
    if (!grouped[row.dataset]) grouped[row.dataset] = {};
    grouped[row.dataset][method] = row;
  });

  const entryMap = grouped[state.selectedDataset] || {};
  const chartDatasets = [state.selectedDataset];
  const forecastingValues = {};
  const classificationValues = {};
  methods.forEach(method => {
    forecastingValues[method] = [getForecastingMetric(entryMap[method])];
    classificationValues[method] = [getClassificationMetric(entryMap[method])];
  });

  renderBenchmarkDatasetChart(
    'rw-forecasting-chart',
    `${state.selectedDataset} - Forecasting RMSE (lower is better)`,
    chartDatasets,
    methods,
    forecastingValues,
    'forecast_rmse',
  );
  renderBenchmarkDatasetChart(
    'rw-classification-chart',
    `${state.selectedDataset} - Classification Accuracy (higher is better)`,
    chartDatasets,
    methods,
    classificationValues,
    'classification_acc',
  );

  if (forecastTitle) forecastTitle.textContent = `${state.selectedDataset} - Forecasting RMSE`;
  if (classificationTitle) classificationTitle.textContent = `${state.selectedDataset} - Classification Accuracy`;
  if (summaryEmpty) summaryEmpty.style.display = 'none';
  if (summaryCharts) summaryCharts.style.display = 'grid';
  if (summaryMeta) {
    const generatedAtText = state.generatedAt ? `Generated at ${state.generatedAt}` : 'Generated time unavailable';
    summaryMeta.textContent = `${generatedAtText}. Showing dataset-level method comparison (${filteredDatasets.length}/${allDatasets.length} visible).`;
  }
}

function bindRealworldBenchmarkControls() {
  if (rwBenchmarkControlsBound) return;
  const { benchmarkSelect, datasetSearchInput, datasetSelect } = getRealworldControlElements();
  if (!benchmarkSelect || !datasetSearchInput || !datasetSelect) return;

  benchmarkSelect.addEventListener('change', async () => {
    const selectedFile = benchmarkSelect.value || '';
    if (!selectedFile || selectedFile === AppState.dashboardBenchmark.benchmarkFile) return;
    const requestId = ++rwBenchmarkRequestId;
    benchmarkSelect.disabled = true;
    try {
      const payload = await fetchDashboardBenchmark(selectedFile);
      if (requestId !== rwBenchmarkRequestId) return;
      renderRealworldBenchmarks(payload, { preserveSelection: false });
    } catch (error) {
      console.error('Failed to load selected benchmark run:', error);
    } finally {
      benchmarkSelect.disabled = false;
    }
  });

  datasetSearchInput.addEventListener('input', () => {
    AppState.dashboardBenchmark.datasetSearch = datasetSearchInput.value || '';
    renderSelectedRealworldDataset();
  });

  datasetSelect.addEventListener('change', () => {
    AppState.dashboardBenchmark.selectedDataset = datasetSelect.value || '';
    renderSelectedRealworldDataset();
  });

  rwBenchmarkControlsBound = true;
}

function renderRealworldBenchmarks(benchmarkData, options = {}) {
  const { benchmarkSelect } = getRealworldControlElements();
  bindRealworldBenchmarkControls();

  const preserveSelection = options.preserveSelection !== false;
  const isAvailable = benchmarkData && benchmarkData.available;
  const rowsRaw = benchmarkData && Array.isArray(benchmarkData.rows) ? benchmarkData.rows : [];
  const rows = rowsRaw.map(row => ({
    ...row,
    method: normalizePrimaryMethodKey(row?.method),
  }));

  if (!isAvailable || rows.length === 0) {
    showRealworldBenchmarkEmpty('Run the RealWorld benchmark to populate dataset-level charts.');
    return;
  }

  const state = AppState.dashboardBenchmark;
  const prevDataset = state.selectedDataset;
  state.availableBenchmarks = Array.isArray(benchmarkData.available_benchmarks)
    ? benchmarkData.available_benchmarks
    : [];
  state.benchmarkFile = benchmarkData.benchmark_file || '';
  state.generatedAt = benchmarkData.generated_at || '';
  state.methods = getMethodListFromBenchmark(benchmarkData);
  state.rows = rows;
  if (!preserveSelection) {
    state.selectedDataset = '';
  }
  if (preserveSelection && prevDataset) {
    state.selectedDataset = prevDataset;
  }

  if (benchmarkSelect) {
    const runOptions = state.availableBenchmarks.map(item => ({
      value: item.file || '',
      label: formatBenchmarkRunLabel(item),
    })).filter(item => item.value);
    setSelectOptions(
      benchmarkSelect,
      runOptions,
      state.benchmarkFile,
      state.generatedAt ? state.generatedAt : 'Current run',
    );
    if (runOptions.length <= 1) {
      benchmarkSelect.disabled = runOptions.length === 0;
    }
  }

  renderSelectedRealworldDataset();
}

const barValueLabelsPlugin = {
  id: 'barValueLabels',
  afterDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    if (!chartArea) return;

    ctx.save();
    ctx.fillStyle = '#475569';
    ctx.font = '10px Arial, "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    chart.data.datasets.forEach((dataset, datasetIndex) => {
      const meta = chart.getDatasetMeta(datasetIndex);
      if (!meta || meta.hidden) return;
      meta.data.forEach((bar, index) => {
        const raw = dataset.data[index];
        if (raw === null || !Number.isFinite(raw)) return;

        const x = bar.x;
        const metricType = chart?.options?.plugins?.barValueLabels?.metricType || 'number';
        const valueLabel = formatMetricValue(raw, metricType);
        let y;

        if (raw >= 0) {
          y = Math.min(bar.base + 6, chartArea.bottom - 2);
        } else {
          y = Math.min(bar.y + 6, chartArea.bottom - 2);
        }

        if (y >= chartArea.top && y <= chartArea.bottom) {
          ctx.fillText(valueLabel, x, y);
        }
      });
    });

    ctx.restore();
  },
};

function renderBenchmarkDatasetChart(canvasId, label, datasets, methods, valuesByMethod, metricType) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const fallbackEl = document.getElementById(`${canvasId}-fallback`);

  const showFallback = (reason) => {
    if (canvas) canvas.style.display = 'none';
    if (fallbackEl) {
      fallbackEl.style.display = 'block';
      fallbackEl.innerHTML = renderBenchmarkFallbackTable(datasets, methods, valuesByMethod, metricType);
    }
    console.error(`Benchmark chart fallback for ${canvasId}: ${reason}`);
  };

  const hideFallback = () => {
    if (canvas) canvas.style.display = 'block';
    if (fallbackEl) {
      fallbackEl.style.display = 'none';
      fallbackEl.innerHTML = '';
    }
  };

  if (typeof Chart === 'undefined') {
    showFallback('Chart.js not loaded');
    return;
  }

  const ctx = canvas;

  const fullLabels = datasets.slice();
  const shortLabels = datasets.map(name => truncateLabel(name, 12));

  const palette = {
    'ImputePilot': '#2563eb',
    'FLAML': '#22c55e',
    'Tune': '#f97316',
    'AutoFolio': '#a855f7',
    'RAHA': '#0ea5e9',
  };
  const fallbackColors = ['#2563eb', '#22c55e', '#f97316', '#a855f7', '#0ea5e9'];

  const datasetsConfig = methods.map((method, idx) => {
    const values = valuesByMethod[method] || [];
    const color = palette[method] || fallbackColors[idx % fallbackColors.length];
    return {
      label: displayMethodName(method),
      data: values,
      backgroundColor: color,
      borderRadius: 6,
      maxBarThickness: 24,
    };
  });

  const flatValues = [];
  datasetsConfig.forEach(dataset => {
    dataset.data.forEach(value => {
      if (value !== null && Number.isFinite(value)) flatValues.push(value);
    });
  });
  const minValue = flatValues.length ? Math.min(...flatValues) : 0;
  const maxValue = flatValues.length ? Math.max(...flatValues) : 1;
  const spreadPadding = flatValues.length ? (maxValue - minValue) * 0.2 : 0.1;
  const fallbackPadding = metricType === 'classification_acc'
    ? 0.05
    : Math.max(Math.abs(maxValue || 1) * 0.1, 0.1);
  const padding = Math.max(spreadPadding, fallbackPadding);
  const suggestedMinRaw = minValue - padding;
  const suggestedMaxRaw = maxValue + padding;
  const suggestedMin = metricType === 'classification_acc' ? Math.max(0, suggestedMinRaw) : suggestedMinRaw;
  const suggestedMax = metricType === 'classification_acc' ? Math.min(1, suggestedMaxRaw) : suggestedMaxRaw;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: { left: 8, right: 8, bottom: 20 }
    },
    plugins: {
      legend: { display: true, position: 'top' },
      tooltip: {
        callbacks: {
          title: (items) => {
            if (!items.length) return '';
            return fullLabels[items[0].dataIndex] || items[0].label || '';
          },
          label: (context) => {
            const raw = context.raw;
            if (raw === null || !Number.isFinite(raw)) {
              return `${context.dataset.label}: No data`;
            }
            return `${context.dataset.label}: ${formatMetricValue(raw, metricType)}`;
          },
        },
      },
      barValueLabels: {
        metricType,
      },
    },
    scales: {
      x: {
        offset: true,
        ticks: {
          color: '#475569',
          maxRotation: 0,
          minRotation: 0,
          autoSkip: false,
          padding: 6,
          font: { size: 10 },
        },
        grid: { display: false },
      },
      y: {
        suggestedMin,
        suggestedMax,
        beginAtZero: false,
        ticks: {
          color: '#64748b',
          callback: value => formatMetricValue(value, metricType),
        },
        grid: { color: '#e5e7eb' },
      },
    },
  };

  try {
    hideFallback();
    if (metricType === 'forecast_rmse') {
      if (rwForecastChart) rwForecastChart.destroy();
      rwForecastChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: shortLabels,
          datasets: datasetsConfig,
        },
        options: chartOptions,
        plugins: [barValueLabelsPlugin],
      });
    } else {
      if (rwClassificationChart) rwClassificationChart.destroy();
      rwClassificationChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: shortLabels,
          datasets: datasetsConfig,
        },
        options: chartOptions,
        plugins: [barValueLabelsPlugin],
      });
    }
  } catch (error) {
    showFallback(error && error.message ? error.message : error);
  }
}

function renderBenchmarkFallbackTable(datasets, methods, valuesByMethod, metricType) {
  const header = ['Dataset'].concat(methods.map(displayMethodName));
  const rows = datasets.map((dataset, rowIndex) => {
    const cols = methods.map((method) => {
      const values = valuesByMethod[method] || [];
      return formatMetricValue(values[rowIndex], metricType);
    });
    return [dataset].concat(cols);
  });

  return `
    <table>
      <thead>
        <tr>${header.map(col => `<th>${col}</th>`).join('')}</tr>
      </thead>
      <tbody>
        ${rows.map(row => `<tr>${row.map(col => `<td>${col}</td>`).join('')}</tr>`).join('')}
      </tbody>
    </table>
  `;
}

function truncateLabel(label, maxLength) {
  if (!label || label.length <= maxLength) return label;
  return `${label.slice(0, maxLength - 1)}…`;
}

function formatMetricValue(value, metricType) {
  if (value === null || !Number.isFinite(value)) {
    return '--';
  }
  if (metricType === 'forecast_rmse') {
    return Number(value).toFixed(3);
  }
  if (metricType === 'classification_acc') {
    return Number(value).toFixed(3);
  }
  return Number(value).toFixed(2);
}

function toNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function getForecastingMetric(entry) {
  if (!entry || entry.status !== 'success') return null;
  return toNumber(entry.forecasting_rmse);
}

function getClassificationMetric(entry) {
  if (!entry || entry.status !== 'success') return null;
  return toNumber(entry.classification_acc);
}

function destroyRealworldCharts() {
  if (rwForecastChart) {
    rwForecastChart.destroy();
    rwForecastChart = null;
  }
  if (rwClassificationChart) {
    rwClassificationChart.destroy();
    rwClassificationChart = null;
  }
}

// Load dashboard data on initial page load
document.addEventListener('DOMContentLoaded', () => {
  loadDashboardData();
});

// ========== Stepper Update ==========
function updatePipelineClusterLayoutState(isActive) {
  const pipelinePage = document.getElementById('page-pipeline');
  if (pipelinePage) {
    pipelinePage.classList.toggle('cluster-layout-active', Boolean(isActive));
  }
  updateMainLayoutLockState();
}

function updateRecommendFeatureLayoutState(isActive) {
  const recommendPage = document.getElementById('page-recommend');
  if (recommendPage) {
    recommendPage.classList.toggle('feature-layout-active', Boolean(isActive));
  }
  updateMainLayoutLockState();
}

function updateMainLayoutLockState() {
  const mainEl = document.querySelector('.main');
  if (!mainEl) return;

  const pipelinePage = document.getElementById('page-pipeline');
  const recommendPage = document.getElementById('page-recommend');
  const pipelineLocked = Boolean(
    pipelinePage &&
    pipelinePage.classList.contains('active') &&
    pipelinePage.classList.contains('cluster-layout-active')
  );
  const recommendLocked = Boolean(
    recommendPage &&
    recommendPage.classList.contains('active') &&
    recommendPage.classList.contains('feature-layout-active')
  );

  mainEl.classList.toggle('cluster-layout-lock', pipelineLocked || recommendLocked);
}

function updatePipelineStepper(step) {
  AppState.pipelineStep = step;
  updatePipelineClusterLayoutState([2, 3, 4, 5].includes(step));

  document.querySelectorAll('#page-pipeline .stepper .step').forEach((el, i) => {
    el.classList.remove('active', 'completed');
    if (i + 1 < step) el.classList.add('completed');
    else if (i + 1 === step) el.classList.add('active');
  });

  document.querySelectorAll('#page-pipeline .step-content').forEach(el => {
    el.classList.remove('active');
    if (parseInt(el.dataset.step) === step) el.classList.add('active');
  });

  if (step === 3) {
    requestAnimationFrame(() => syncClusterPanelsHeight());
  }
}

function syncClusterPanelsHeight() {
  const configPanel = document.getElementById('cluster-config-panel');
  const resultPanel = document.getElementById('cluster-result-panel');
  const resultsContainer = document.getElementById('cluster-results');
  if (!configPanel || !resultPanel) return;

  if (window.innerWidth <= 1024) {
    resultPanel.style.height = '';
    resultPanel.style.minHeight = '';
    if (resultsContainer) resultsContainer.style.maxHeight = '';
    return;
  }

  const configHeight = Math.ceil(configPanel.getBoundingClientRect().height);
  if (!configHeight) return;

  resultPanel.style.height = `${configHeight}px`;
  resultPanel.style.minHeight = `${configHeight}px`;
  if (resultsContainer) {
    resultsContainer.style.maxHeight = '';
  }
}

function updateRecommendStepper(step) {
  AppState.recommendStep = step;
  updateRecommendFeatureLayoutState(step === 2);

  document.querySelectorAll('#page-recommend .stepper .step').forEach((el, i) => {
    el.classList.remove('active', 'completed');
    if (i + 1 < step) el.classList.add('completed');
    else if (i + 1 === step) el.classList.add('active');
  });

  document.querySelectorAll('#page-recommend .step-content').forEach(el => {
    el.classList.remove('active');
    if (parseInt(el.dataset.step) === step) el.classList.add('active');
  });
}

// ========== Console Logging ==========
function logPipeline(msg, type = 'info') {
  logToConsole('pipeline-console', msg, type);
}

function logRecommend(msg, type = 'info') {
  logToConsole('recommend-console', msg, type);
}

function logToConsole(consoleId, msg, type) {
  const consoleEl = document.getElementById(consoleId);
  if (!consoleEl) return;

  const time = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = `console-line ${type}`;
  line.textContent = `[${time}] ${normalizeBrandText(String(msg))}`;
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

// ========== File Upload - Pipeline ==========
const pipelineUploadZone = document.getElementById('pipeline-upload-zone');
const pipelineFileInput = document.getElementById('pipeline-file-input');
const pipelineFileList = document.getElementById('pipeline-file-list');
const btnPipelineContinue = document.getElementById('btn-pipeline-continue');

if (pipelineUploadZone) {
  pipelineUploadZone.addEventListener('click', () => pipelineFileInput.click());

  pipelineFileInput.addEventListener('change', (e) => {
    handlePipelineFiles(e.target.files);
  });

  pipelineUploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    pipelineUploadZone.classList.add('dragover');
  });

  pipelineUploadZone.addEventListener('dragleave', () => {
    pipelineUploadZone.classList.remove('dragover');
  });

  pipelineUploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    pipelineUploadZone.classList.remove('dragover');
    handlePipelineFiles(e.dataTransfer.files);
  });
}

function handlePipelineFiles(files) {
  const valid = Array.from(files).filter(f =>
    f.name.endsWith('.zip')
  );

  if (valid.length === 0) {
    logPipeline('Please upload .zip files', 'error');
    return;
  }

  let added = 0;
  valid.forEach(file => {
    if (!AppState.pipelineFiles.find(f => f.name === file.name)) {
      AppState.pipelineFiles.push(file);
      added += 1;
    }
  });

  if (added === 0) {
    logPipeline('Selected file(s) already added', 'info');
    return;
  }

  AppState.pipelineUploadResult = null;
  AppState.pipelineDatasetId = null;
  uploadCompleted = false;

  renderPipelineFileList();
  updatePipelineUI();
  logPipeline(`${added} file(s) added`, 'success');
  uploadPipelineForPreview();
}

function renderPipelineFileList() {
  pipelineFileList.innerHTML = AppState.pipelineFiles.map((file, i) => `
    <div class="file-item">
      <div class="file-info">
        <div class="file-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div>
          <div class="file-name">${file.name}</div>
          <div class="file-size">${formatSize(file.size)}</div>
        </div>
      </div>
      <button class="file-remove" onclick="removePipelineFile(${i})">✕</button>
    </div>
  `).join('');
}

function removePipelineFile(index) {
  AppState.pipelineFiles.splice(index, 1);
  pipelinePreviewRequestId += 1;
  AppState.pipelineUploadResult = null;
  AppState.pipelineDatasetId = null;
  uploadCompleted = false;
  renderPipelineFileList();
  updatePipelineUI();
  if (AppState.pipelineFiles.length === 0) {
    document.getElementById('pipeline-preview-placeholder').style.display = 'flex';
    const placeholderText = document.querySelector('#pipeline-preview-placeholder span');
    if (placeholderText) {
      placeholderText.textContent = 'Upload data to begin';
    }
    document.getElementById('pipeline-preview-content').style.display = 'none';
  } else {
    uploadPipelineForPreview();
  }
}

function updatePipelineUI() {
  if (AppState.pipelineFiles.length > 0) {
    pipelineUploadZone.classList.add('has-files');
    btnPipelineContinue.disabled = false;
    btnPipelineContinue.textContent = uploadCompleted ? 'Continue to Feature Extraction →' : 'Continue to feature extraction';
  } else {
    pipelineUploadZone.classList.remove('has-files');
    btnPipelineContinue.disabled = true;
    btnPipelineContinue.textContent = 'Continue to feature extraction';
  }
}

// Global variable for chart instance
let pipelineChart = null;
let rwForecastChart = null;
let rwClassificationChart = null;
let rwBenchmarkControlsBound = false;
let rwBenchmarkRequestId = 0;
let pipelinePreviewRequestId = 0;

function setPipelinePreviewPlaceholder(text) {
  const placeholder = document.getElementById('pipeline-preview-placeholder');
  const placeholderText = document.querySelector('#pipeline-preview-placeholder span');
  const content = document.getElementById('pipeline-preview-content');
  if (content) content.style.display = 'none';
  if (placeholder) placeholder.style.display = 'flex';
  if (placeholderText && text) placeholderText.textContent = text;
}

async function uploadPipelineForPreview() {
  if (!AppState.pipelineFiles.length) {
    setPipelinePreviewPlaceholder('Upload data to begin');
    return;
  }

  const requestId = ++pipelinePreviewRequestId;
  setPipelinePreviewPlaceholder('Uploading file and generating preview...');
  btnPipelineContinue.disabled = true;

  try {
    const result = await uploadFiles(API_CONFIG.endpoints.uploadTraining, AppState.pipelineFiles);
    if (requestId !== pipelinePreviewRequestId) return;

    AppState.pipelineUploadResult = result || null;
    AppState.pipelineDatasetId = result?.datasetId || null;
    uploadCompleted = Boolean(result?.datasetId);

    if (result?.preview) {
      previewPipelineFile(result.preview);
      logPipeline('Preview generated from uploaded file.', 'success');
    } else {
      setPipelinePreviewPlaceholder('Preview unavailable for this file');
      logPipeline('Preview unavailable for uploaded file.', 'warn');
    }
  } catch (error) {
    if (requestId !== pipelinePreviewRequestId) return;
    AppState.pipelineUploadResult = null;
    AppState.pipelineDatasetId = null;
    uploadCompleted = false;
    setPipelinePreviewPlaceholder('Preview failed. Please upload again.');
    logPipeline(`Upload/preview failed: ${error.message}`, 'error');
  } finally {
    if (requestId === pipelinePreviewRequestId) {
      updatePipelineUI();
    }
  }
}

function previewPipelineFile(preview) {
  console.log('[DEBUG] previewPipelineFile called with:', preview);

  if (!preview) {
    document.getElementById('pipeline-preview-placeholder').style.display = 'flex';
    document.getElementById('pipeline-preview-content').style.display = 'none';
    return;
  }

  document.getElementById('pipeline-preview-placeholder').style.display = 'none';
  document.getElementById('pipeline-preview-content').style.display = 'block';

  const seriesList = buildPipelinePreviewSeries(preview);
  const statsEl = document.getElementById('pipeline-preview-stats');
  if (!statsEl) return;
  const totalSeries = Number(preview?.totalRows) || seriesList.length;
  const previewSeriesCount = Math.min(seriesList.length, totalSeries);

  if (!seriesList.length) {
    statsEl.innerHTML = `
      <div class="pipeline-stat-card"><span class="pipeline-stat-value">No series available</span></div>
      <div class="pipeline-stat-card"><span class="pipeline-stat-value">First 0 / 0 time series</span></div>
      <div class="pipeline-stat-card"><span class="pipeline-stat-value">0 time points</span></div>
    `;
    renderPipelineMissingInfo({ totalPoints: 0, missingPoints: 0, missingRate: 0 });
    if (pipelineChart) {
      pipelineChart.destroy();
      pipelineChart = null;
    }
    return;
  }

  const seriesOptions = seriesList
    .map((series, idx) => `<option value="${idx}">${series.label}</option>`)
    .join('');

  statsEl.innerHTML = `
    <div class="pipeline-stat-card pipeline-stat-selector">
      <label for="pipeline-series-select">Time Series</label>
      <select id="pipeline-series-select" class="pipeline-series-select">
        ${seriesOptions}
      </select>
    </div>
    <div class="pipeline-stat-card">
      <span class="pipeline-stat-value" id="pipeline-preview-series-count"></span>
    </div>
    <div class="pipeline-stat-card">
      <span class="pipeline-stat-value" id="pipeline-selected-points-count"></span>
    </div>
  `;

  const selectEl = document.getElementById('pipeline-series-select');
  const seriesCountEl = document.getElementById('pipeline-preview-series-count');
  const pointsCountEl = document.getElementById('pipeline-selected-points-count');

  if (seriesCountEl) {
    seriesCountEl.textContent = `First ${previewSeriesCount} / ${totalSeries} time series`;
  }

  const renderSelectedSeries = (rawIdx) => {
    const idx = Number.isInteger(rawIdx) ? rawIdx : parseInt(rawIdx, 10);
    const safeIdx = Number.isFinite(idx)
      ? Math.min(Math.max(idx, 0), seriesList.length - 1)
      : 0;
    const selected = seriesList[safeIdx];
    if (pointsCountEl) {
      pointsCountEl.textContent = `${selected.totalPoints} time points`;
    }

    renderPipelineMissingInfo(selected);
    renderTimeSeriesChart(selected.chartData);
  };

  if (selectEl) {
    selectEl.addEventListener('change', (event) => {
      renderSelectedSeries(event.target.value);
    });
  }

  renderSelectedSeries(0);

  console.log('[DEBUG] Preview rendered successfully');
}

function normalizePipelinePreviewPoint(value, idx) {
  if (value && typeof value === 'object') {
    const xRaw = value.x !== undefined ? value.x : idx;
    const yRaw = value.y;
    const missingRaw = value.missing === true;
    const x = Number.isFinite(Number(xRaw)) ? Number(xRaw) : idx;
    const y = Number.isFinite(Number(yRaw)) ? Number(yRaw) : null;
    const missing = missingRaw || y === null;
    return { x, y: missing ? null : y, missing };
  }

  const text = value === null || value === undefined ? '' : String(value).trim();
  const lowerText = text.toLowerCase();
  if (!text || lowerText === 'nan' || lowerText === 'null' || lowerText === 'none') {
    return { x: idx, y: null, missing: true };
  }

  const y = Number(text);
  if (!Number.isFinite(y)) {
    return { x: idx, y: null, missing: true };
  }
  return { x: idx, y, missing: false };
}

function buildPipelinePreviewSeries(preview) {
  const seriesRows = Array.isArray(preview?.seriesRows) && preview.seriesRows.length
    ? preview.seriesRows
    : [];

  const fallbackRows = !seriesRows.length && Array.isArray(preview?.headers) && preview.headers.length
    ? [preview.headers, ...(Array.isArray(preview?.rows) ? preview.rows : [])]
    : [];

  const sourceRows = (seriesRows.length ? seriesRows : fallbackRows)
    .filter((row) => Array.isArray(row) && row.length);

  if (!sourceRows.length && Array.isArray(preview?.chartData) && preview.chartData.length) {
    const chartData = preview.chartData.map((point, idx) => normalizePipelinePreviewPoint(point, idx));
    const missingPoints = chartData.filter((point) => point.missing).length;
    const totalPoints = chartData.length;
    return [{
      label: 'Series 1',
      chartData,
      totalPoints,
      missingPoints,
      missingRate: totalPoints > 0 ? ((missingPoints / totalPoints) * 100).toFixed(1) : '0.0',
    }];
  }

  return sourceRows.map((row, idx) => {
    const chartData = row.map((value, pointIdx) => normalizePipelinePreviewPoint(value, pointIdx));
    const totalPoints = chartData.length;
    const missingPoints = chartData.filter((point) => point.missing).length;
    const missingRate = totalPoints > 0 ? ((missingPoints / totalPoints) * 100).toFixed(1) : '0.0';
    return {
      label: `Series ${idx + 1}`,
      chartData,
      totalPoints,
      missingPoints,
      missingRate,
    };
  });
}

function renderPipelineMissingInfo(seriesMeta) {
  const missingInfoEl = document.getElementById('pipeline-missing-info');
  if (!missingInfoEl || !seriesMeta) return;

  const totalPoints = Number(seriesMeta.totalPoints) || 0;
  const missingPoints = Number(seriesMeta.missingPoints) || 0;
  const validPoints = Math.max(totalPoints - missingPoints, 0);
  const missingRate = Number(seriesMeta.missingRate);
  const rateText = Number.isFinite(missingRate) ? missingRate.toFixed(1) : '0.0';

  missingInfoEl.innerHTML = `
    <div class="info-item">
      <span class="legend-dot valid"></span>
      <span class="stat-label">Valid Points:</span>
      <span class="stat-value">${validPoints}</span>
    </div>
    <div class="info-item">
      <span class="legend-dot missing"></span>
      <span class="stat-label">Missing Points:</span>
      <span class="stat-value">${missingPoints} (${rateText}%)</span>
    </div>
  `;
}

// Render time series line chart with missing regions highlighted
function renderTimeSeriesChart(chartData) {
  const ctx = document.getElementById('pipeline-ts-chart');
  if (!ctx) {
    console.error('[ERROR] Chart canvas not found!');
    return;
  }

  // Destroy old chart
  if (pipelineChart) {
    pipelineChart.destroy();
  }

  // Identify missing regions
  const missingRegions = [];
  let missingStart = null;

  chartData.forEach((point, idx) => {
    if (point.missing) {
      if (missingStart === null) {
        missingStart = point.x;
      }
    } else {
      if (missingStart !== null) {
        missingRegions.push({ start: missingStart, end: point.x - 1 });
        missingStart = null;
      }
    }
  });

  // Handle trailing missing region
  if (missingStart !== null) {
    missingRegions.push({ start: missingStart, end: chartData.length - 1 });
  }

  // Prepare data for Chart.js
  const labels = chartData.map(p => p.x);
  const values = chartData.map(p => p.missing ? null : p.y);

  // Calculate Y axis range
  const validValues = values.filter(v => v !== null);
  const minY = validValues.length > 0 ? Math.min(...validValues) : 0;
  const maxY = validValues.length > 0 ? Math.max(...validValues) : 1;
  const padding = (maxY - minY) * 0.1 || 0.1;

  pipelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Time Series Values',
          data: values,
          borderColor: '#4CAF50',
          backgroundColor: 'rgba(76, 175, 80, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4,
          spanGaps: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            title: function (context) {
              return `Time Point: ${context[0].label}`;
            },
            label: function (context) {
              if (context.raw === null) {
                return 'Missing Value';
              }
              return `Value: ${context.raw.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'Time Point',
            font: { size: 11 }
          },
          ticks: {
            maxTicksLimit: 10,
            font: { size: 10 }
          }
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Value',
            font: { size: 11 }
          },
          min: minY - padding,
          max: maxY + padding,
          ticks: {
            font: { size: 10 }
          }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    },
    plugins: [{
      // Custom plugin to draw missing region backgrounds
      id: 'missingRegions',
      beforeDraw: function (chart) {
        const ctx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;

        ctx.save();
        missingRegions.forEach(region => {
          const xStart = xAxis.getPixelForValue(region.start);
          const xEnd = xAxis.getPixelForValue(region.end);
          const yTop = yAxis.top;
          const yBottom = yAxis.bottom;

          // Fill missing region
          ctx.fillStyle = 'rgba(255, 107, 107, 0.15)';
          ctx.fillRect(xStart, yTop, xEnd - xStart + 5, yBottom - yTop);

          // Draw border
          ctx.strokeStyle = 'rgba(255, 107, 107, 0.5)';
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 5]);
          ctx.strokeRect(xStart, yTop, xEnd - xStart + 5, yBottom - yTop);
          ctx.setLineDash([]);
        });
        ctx.restore();
      }
    }]
  });

  console.log('[DEBUG] Chart rendered with', missingRegions.length, 'missing regions');
}

// ========== Recommend Preview ==========
let recommendChart = null;
let imputedChart = null;
let recommendSeriesParseRequestId = 0;

function setRecommendPreviewPlaceholder(text) {
  const placeholder = document.getElementById('recommend-preview-placeholder');
  const placeholderText = document.getElementById('recommend-preview-placeholder-text');
  const content = document.getElementById('recommend-preview-content');
  if (content) content.style.display = 'none';
  if (placeholder) placeholder.style.display = 'flex';
  if (placeholderText && text) placeholderText.textContent = text;
}

function previewRecommendFile(preview) {
  if (!preview) {
    setRecommendPreviewPlaceholder('Upload data to see preview');
    return;
  }

  const placeholder = document.getElementById('recommend-preview-placeholder');
  const content = document.getElementById('recommend-preview-content');
  if (placeholder) placeholder.style.display = 'none';
  if (content) content.style.display = 'block';

  const statsEl = document.getElementById('recommend-preview-stats');
  if (statsEl) {
    statsEl.innerHTML = `
      <span class="stat-badge">${preview.fileName}</span>
      <span class="stat-badge">${preview.totalRows} time series</span>
      <span class="stat-badge">${preview.columns} time points</span>
    `;
  }

  const missingInfoEl = document.getElementById('recommend-missing-info');
  if (missingInfoEl && preview.missingPoints !== undefined) {
    missingInfoEl.innerHTML = `
      <div class="info-item">
        <span class="legend-dot valid"></span>
        <span class="stat-label">Valid Points:</span>
        <span class="stat-value">${preview.totalPoints - preview.missingPoints}</span>
      </div>
      <div class="info-item">
        <span class="legend-dot missing"></span>
        <span class="stat-label">Missing Points:</span>
        <span class="stat-value">${preview.missingPoints} (${preview.missingRate}%)</span>
      </div>
    `;
  }

  if (preview.chartData && preview.chartData.length > 0) {
    renderRecommendTimeSeriesChart(preview.chartData);
  }

  if (!Array.isArray(AppState.recommendSeriesList) || AppState.recommendSeriesList.length === 0) {
    const previewSeriesList = buildPipelinePreviewSeries(preview);
    if (previewSeriesList.length > 0) {
      AppState.recommendSeriesList = previewSeriesList;
      AppState.selectedRecoveredSeriesIndex = 0;
      refreshRecoveredSeriesSelector();
    }
  }
}

function renderRecommendTimeSeriesChart(chartData) {
  const ctx = document.getElementById('recommend-ts-chart');
  if (!ctx) return;

  if (recommendChart) {
    recommendChart.destroy();
  }

  const missingRegions = [];
  let missingStart = null;
  chartData.forEach((point, idx) => {
    if (point.missing) {
      if (missingStart === null) missingStart = point.x;
    } else if (missingStart !== null) {
      missingRegions.push({ start: missingStart, end: point.x - 1 });
      missingStart = null;
    }
  });
  if (missingStart !== null) {
    missingRegions.push({ start: missingStart, end: chartData.length - 1 });
  }

  const labels = chartData.map(p => p.x);
  const values = chartData.map(p => p.missing ? null : p.y);
  const validValues = values.filter(v => v !== null);
  const minY = validValues.length > 0 ? Math.min(...validValues) : 0;
  const maxY = validValues.length > 0 ? Math.max(...validValues) : 1;
  const padding = (maxY - minY) * 0.1 || 0.1;

  recommendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Time Series Values',
          data: values,
          borderColor: '#4CAF50',
          backgroundColor: 'rgba(76, 175, 80, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4,
          spanGaps: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            title: function (context) {
              return `Time Point: ${context[0].label}`;
            },
            label: function (context) {
              if (context.raw === null) {
                return 'Missing Value';
              }
              return `Value: ${context.raw.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Time Point', font: { size: 11 } },
          ticks: { maxTicksLimit: 10, font: { size: 10 } }
        },
        y: {
          display: true,
          title: { display: true, text: 'Value', font: { size: 11 } },
          min: minY - padding,
          max: maxY + padding,
          ticks: { font: { size: 10 } }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    },
    plugins: [{
      id: 'recommendMissingRegions',
      beforeDraw: function (chart) {
        const chartCtx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        chartCtx.save();
        missingRegions.forEach(region => {
          const xStart = xAxis.getPixelForValue(region.start);
          const xEnd = xAxis.getPixelForValue(region.end);
          const yTop = yAxis.top;
          const yBottom = yAxis.bottom;
          chartCtx.fillStyle = 'rgba(255, 107, 107, 0.15)';
          chartCtx.fillRect(xStart, yTop, xEnd - xStart + 5, yBottom - yTop);
          chartCtx.strokeStyle = 'rgba(255, 107, 107, 0.5)';
          chartCtx.lineWidth = 1;
          chartCtx.setLineDash([5, 5]);
          chartCtx.strokeRect(xStart, yTop, xEnd - xStart + 5, yBottom - yTop);
          chartCtx.setLineDash([]);
        });
        chartCtx.restore();
      }
    }]
  });
}

function buildImputedValuesFallback(chartData) {
  const values = chartData.map(p => (p && p.y !== undefined ? p.y : null));
  const filled = [...values];
  const knownIdx = values
    .map((v, i) => (v !== null && v !== undefined ? i : null))
    .filter(i => i !== null);

  if (knownIdx.length === 0) return filled;

  const firstKnown = knownIdx[0];
  for (let i = 0; i < firstKnown; i += 1) {
    filled[i] = values[firstKnown];
  }

  for (let k = 0; k < knownIdx.length - 1; k += 1) {
    const start = knownIdx[k];
    const end = knownIdx[k + 1];
    const startVal = values[start];
    const endVal = values[end];
    const gap = end - start;
    if (gap > 1) {
      for (let i = start + 1; i < end; i += 1) {
        const t = (i - start) / gap;
        filled[i] = startVal + (endVal - startVal) * t;
      }
    }
  }

  const lastKnown = knownIdx[knownIdx.length - 1];
  for (let i = lastKnown + 1; i < filled.length; i += 1) {
    filled[i] = values[lastKnown];
  }

  return filled;
}

function refreshRecoveredSeriesSelector() {
  const controls = document.getElementById('recovered-series-controls');
  const selectEl = document.getElementById('recovered-series-select');
  const metaEl = document.getElementById('recovered-series-meta');
  const seriesList = Array.isArray(AppState.recommendSeriesList) ? AppState.recommendSeriesList : [];

  if (!controls || !selectEl || !metaEl) return;

  if (!seriesList.length) {
    controls.style.display = 'none';
    selectEl.innerHTML = '';
    metaEl.textContent = '';
    return;
  }

  const currentIdx = Number(AppState.selectedRecoveredSeriesIndex);
  const safeIdx = Number.isFinite(currentIdx)
    ? Math.min(Math.max(currentIdx, 0), seriesList.length - 1)
    : 0;
  AppState.selectedRecoveredSeriesIndex = safeIdx;

  const optionHtml = seriesList
    .map((series, idx) => `<option value="${idx}">${series.label || `Series ${idx + 1}`}</option>`)
    .join('');
  if (selectEl.innerHTML !== optionHtml) {
    selectEl.innerHTML = optionHtml;
  }
  selectEl.value = String(safeIdx);

  const selectedSeries = seriesList[safeIdx];
  const totalPoints = Number(selectedSeries?.totalPoints) || 0;
  const missingPoints = Number(selectedSeries?.missingPoints) || 0;
  const missingRate = Number(selectedSeries?.missingRate);
  const missingRateText = Number.isFinite(missingRate) ? missingRate.toFixed(1) : '0.0';
  metaEl.textContent = `${safeIdx + 1}/${seriesList.length} · ${totalPoints} points · Missing ${missingPoints} (${missingRateText}%)`;

  controls.style.display = 'flex';
}

async function loadRecommendSeriesFromLocalFile() {
  const file = AppState.recommendFiles?.[0];
  if (!file) return;

  const requestId = ++recommendSeriesParseRequestId;
  if (String(file.name || '').toLowerCase().endsWith('.zip')) return;

  try {
    const text = await file.text();
    if (requestId !== recommendSeriesParseRequestId) return;

    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    if (!lines.length) return;

    const sample = lines[0];
    const delimiter = sample.includes(',')
      ? ','
      : (sample.includes('\t') ? '\t' : ' ');
    const seriesRows = lines.map((line) => {
      if (delimiter === ' ') return line.split(/\s+/);
      return line.split(delimiter);
    }).filter((row) => Array.isArray(row) && row.length > 0);

    const fullSeriesList = buildPipelinePreviewSeries({ seriesRows });
    if (!fullSeriesList.length) return;

    AppState.recommendSeriesList = fullSeriesList;
    const idx = Number(AppState.selectedRecoveredSeriesIndex);
    AppState.selectedRecoveredSeriesIndex = Number.isFinite(idx)
      ? Math.min(Math.max(idx, 0), fullSeriesList.length - 1)
      : 0;
    refreshRecoveredSeriesSelector();

    if (AppState.lastImputationResults) {
      renderImputationResults(AppState.lastImputationResults);
    }
  } catch (error) {
    console.warn('[Recommend] Failed to parse local series file:', error);
  }
}

function renderImputedTimeSeriesChart(chartData, imputedValues = null) {
  const ctx = document.getElementById('recovered-ts-chart');
  if (!ctx) return;

  const placeholder = document.getElementById('recovered-chart-placeholder');
  if (!chartData || chartData.length === 0) {
    if (placeholder) placeholder.classList.remove('hidden');
    return;
  }

  if (placeholder) placeholder.classList.add('hidden');

  if (imputedChart) {
    imputedChart.destroy();
  }

  const missingRegions = [];
  let missingStart = null;
  chartData.forEach((point, idx) => {
    if (point.missing) {
      if (missingStart === null) missingStart = point.x ?? idx;
    } else if (missingStart !== null) {
      missingRegions.push({ start: missingStart, end: (point.x ?? idx) - 1 });
      missingStart = null;
    }
  });
  if (missingStart !== null) {
    missingRegions.push({ start: missingStart, end: chartData.length - 1 });
  }

  const labels = chartData.map((p, i) => p.x ?? i);
  const observedValues = chartData.map(p => (p.missing ? null : p.y));
  const resolvedImputedValues = imputedValues || buildImputedValuesFallback(chartData);
  const imputedPoints = chartData.map((p, i) => (p.missing ? resolvedImputedValues[i] : null));

  const validValues = resolvedImputedValues.filter(v => v !== null);
  const minY = validValues.length > 0 ? Math.min(...validValues) : 0;
  const maxY = validValues.length > 0 ? Math.max(...validValues) : 1;
  const padding = (maxY - minY) * 0.1 || 0.1;

  imputedChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Observed Values',
          data: observedValues,
          borderColor: '#4CAF50',
          backgroundColor: 'rgba(76, 175, 80, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4,
          spanGaps: false,
        },
        {
          label: 'Imputed Line',
          data: resolvedImputedValues,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.08)',
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          spanGaps: true,
        },
        {
          label: 'Imputed Points',
          data: imputedPoints,
          borderColor: '#3b82f6',
          backgroundColor: '#3b82f6',
          borderWidth: 0,
          pointRadius: 3,
          pointHoverRadius: 4,
          showLine: false,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            title: function (context) {
              return `Time Point: ${context[0].label}`;
            },
            label: function (context) {
              if (context.raw === null) {
                return 'Missing Value';
              }
              return `${context.dataset.label}: ${context.raw.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Time Point', font: { size: 11 } },
          ticks: { maxTicksLimit: 10, font: { size: 10 } }
        },
        y: {
          display: true,
          title: { display: true, text: 'Value', font: { size: 11 } },
          min: minY - padding,
          max: maxY + padding,
          ticks: { font: { size: 10 } }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    },
    plugins: [{
      id: 'imputedMissingRegions',
      beforeDraw: function (chart) {
        const chartCtx = chart.ctx;
        const xAxis = chart.scales.x;
        const yAxis = chart.scales.y;
        chartCtx.save();
        missingRegions.forEach(region => {
          const xStart = xAxis.getPixelForValue(region.start);
          const xEnd = xAxis.getPixelForValue(region.end);
          const yTop = yAxis.top;
          const yBottom = yAxis.bottom;
          chartCtx.fillStyle = 'rgba(255, 107, 107, 0.12)';
          chartCtx.fillRect(xStart, yTop, xEnd - xStart + 5, yBottom - yTop);
          chartCtx.strokeStyle = 'rgba(255, 107, 107, 0.4)';
          chartCtx.lineWidth = 1;
          chartCtx.setLineDash([5, 5]);
          chartCtx.strokeRect(xStart, yTop, xEnd - xStart + 5, yBottom - yTop);
          chartCtx.setLineDash([]);
        });
        chartCtx.restore();
      }
    }]
  });
}

// ========== Pipeline Step 1: Upload & Continue ==========
let uploadCompleted = false;
if (btnPipelineContinue) {
  btnPipelineContinue.addEventListener('click', async () => {
    // If upload already completed, continue directly
    if (uploadCompleted && AppState.pipelineDatasetId) {
      updatePipelineStepper(2);
      return;
    }

    try {
      logPipeline('Upload not ready yet, preparing now...', 'info');
      btnPipelineContinue.classList.add('loading');
      btnPipelineContinue.textContent = 'Uploading...';

      // Fallback upload in case user clicks continue before background preview upload finished
      const result = AppState.pipelineUploadResult?.datasetId
        ? AppState.pipelineUploadResult
        : await uploadFiles(API_CONFIG.endpoints.uploadTraining, AppState.pipelineFiles);
      console.log('[DEBUG] Upload result:', result);
      console.log('[DEBUG] Preview data:', result?.preview);

      AppState.pipelineUploadResult = result || null;
      AppState.pipelineDatasetId = result?.datasetId || 'mock-id';

      // Show preview with chart
      if (result?.preview) {
        console.log('[DEBUG] Calling previewPipelineFile');
        previewPipelineFile(result.preview);
      }

      logPipeline('Upload ready. Moving to feature extraction.', 'success');
      btnPipelineContinue.classList.remove('loading');

      uploadCompleted = true;
      updatePipelineUI();
      updatePipelineStepper(2);

    } catch (error) {
      logPipeline(`Upload failed: ${error.message}`, 'error');
      btnPipelineContinue.classList.remove('loading');
      uploadCompleted = false;
      updatePipelineUI();
    }
  });
}

// ========== Pipeline Step 3: Clustering ==========
const btnRunClustering = document.getElementById('btn-run-clustering');
let clusteringCompleted = false;
if (btnRunClustering) {
  btnRunClustering.addEventListener('click', async () => {
    // If clustering already completed, just go to next step
    if (clusteringCompleted) {
      updatePipelineStepper(4);
      return;
    }

    try {
      const delta = parseFloat(document.getElementById('input-delta').value);
      const rho = parseFloat(document.getElementById('input-rho').value);

      logPipeline(`Running clustering (δ=${delta}, ρ=${rho})...`, 'info');
      btnRunClustering.classList.add('loading');
      btnRunClustering.textContent = 'Running...';

      // Call backend API
      let results;
      if (API_CONFIG.useMock) {
        await simulateDelay(2000);
        results = MockData.clusterResults;
      } else {
        const response = await apiCall(API_CONFIG.endpoints.runClustering, 'POST', {
          datasetId: AppState.pipelineDatasetId,
          delta,
          rho,
        });
        results = response.clusters;
      }

      // Store results
      AppState.clusterResults = results;

      // Render results
      renderClusterResults(results);

      logPipeline(`Clustering completed! Found ${results.length} clusters.`, 'success');
      btnRunClustering.classList.remove('loading');

      // Mark as completed and change button text
      clusteringCompleted = true;
      btnRunClustering.textContent = 'Continue to Labeling →';

    } catch (error) {
      logPipeline(`Clustering failed: ${error.message}`, 'error');
      btnRunClustering.classList.remove('loading');
      btnRunClustering.textContent = 'Run Clustering';
    }
  });
}

function renderClusterResults(results) {
  const placeholder = document.getElementById('cluster-placeholder');
  const container = document.getElementById('cluster-results');
  if (!container) return;
  if (placeholder) placeholder.style.display = 'none';
  container.style.display = 'flex';
  container.innerHTML = '';

  const buildSparklineSvg = (points) => {
    const vals = (Array.isArray(points) ? points : [])
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v));
    if (vals.length < 2) return '';

    const w = 560;
    const h = 180;
    const margin = { left: 46, right: 14, top: 12, bottom: 34 };
    const plotW = w - margin.left - margin.right;
    const plotH = h - margin.top - margin.bottom;
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const range = maxV - minV;

    const formatTick = (v) => {
      if (!Number.isFinite(v)) return '--';
      const a = Math.abs(v);
      if (a >= 1000 || (a > 0 && a < 0.001)) return v.toExponential(2);
      return v.toFixed(3).replace(/\.?0+$/, '');
    };

    const coords = vals.map((v, i) => {
      const x = margin.left + (i / (vals.length - 1)) * plotW;
      const yNorm = range === 0 ? 0.5 : (v - minV) / range;
      const y = margin.top + (1 - yNorm) * plotH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    const yTicksCount = 5;
    const yTicks = Array.from({ length: yTicksCount }, (_, i) => {
      const t = i / (yTicksCount - 1);
      const value = maxV - t * (range || 1);
      const y = margin.top + t * plotH;
      return {
        y,
        value: range === 0 ? maxV : value,
      };
    });

    const xTicksCount = Math.min(6, Math.max(3, vals.length));
    const xTicks = Array.from({ length: xTicksCount }, (_, i) => {
      const t = i / (xTicksCount - 1);
      const x = margin.left + t * plotW;
      const idx = Math.round(t * (vals.length - 1));
      return { x, idx };
    });

    return `
      <svg viewBox="0 0 ${w} ${h}" class="cluster-preview-svg" preserveAspectRatio="none" aria-label="Cluster shape preview">
        ${yTicks.map((t) => `
          <line x1="${margin.left}" y1="${t.y.toFixed(2)}" x2="${w - margin.right}" y2="${t.y.toFixed(2)}" class="cluster-preview-grid" />
          <line x1="${(margin.left - 4)}" y1="${t.y.toFixed(2)}" x2="${margin.left}" y2="${t.y.toFixed(2)}" class="cluster-preview-axis" />
          <text x="${margin.left - 7}" y="${(t.y + 4).toFixed(2)}" text-anchor="end" class="cluster-preview-tick-label">${formatTick(t.value)}</text>
        `).join('')}
        ${xTicks.map((t) => `
          <line x1="${t.x.toFixed(2)}" y1="${h - margin.bottom}" x2="${t.x.toFixed(2)}" y2="${h - margin.bottom + 4}" class="cluster-preview-axis" />
          <text x="${t.x.toFixed(2)}" y="${h - margin.bottom + 15}" text-anchor="middle" class="cluster-preview-tick-label">${t.idx}</text>
        `).join('')}
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${h - margin.bottom}" class="cluster-preview-axis" />
        <line x1="${margin.left}" y1="${h - margin.bottom}" x2="${w - margin.right}" y2="${h - margin.bottom}" class="cluster-preview-axis" />
        <polyline points="${coords.join(' ')}" class="cluster-preview-line" />
        <text x="${(w - margin.right).toFixed(2)}" y="${(h - 4)}" text-anchor="end" class="cluster-preview-axis-label">Index</text>
        <text x="${margin.left}" y="${10}" text-anchor="start" class="cluster-preview-axis-label">Value</text>
      </svg>
    `;
  };

  results.forEach((r) => {
    const item = document.createElement('div');
    item.className = 'result-item cluster-result-item';

    const summary = document.createElement('button');
    summary.type = 'button';
    summary.className = 'cluster-summary';
    const rhoVal = Number.isFinite(Number(r.rho)) ? Number(r.rho).toFixed(2) : '--';
    summary.innerHTML = `
      <span class="cluster-name">${r.name}</span>
      <span class="cluster-count">${r.count} series</span>
      <span class="cluster-algo">ρ=${rhoVal}</span>
      <span class="cluster-toggle" aria-hidden="true">▾</span>
    `;

    const previewWrap = document.createElement('div');
    previewWrap.className = 'cluster-preview-wrap';
    const sparkline = buildSparklineSvg(r.shapePreview);
    if (sparkline) {
      const nPoints = Array.isArray(r.shapePreview) ? r.shapePreview.length : 0;
      previewWrap.innerHTML = `
        <div class="cluster-preview-meta">Shape preview (${nPoints} sampled points)</div>
        <div class="cluster-preview-plot">${sparkline}</div>
      `;
    } else {
      previewWrap.innerHTML = `
        <div class="cluster-preview-empty">No shape preview available for this cluster.</div>
      `;
    }

    summary.addEventListener('click', () => {
      item.classList.toggle('expanded');
    });

    item.appendChild(summary);
    item.appendChild(previewWrap);
    container.appendChild(item);
  });

  requestAnimationFrame(() => syncClusterPanelsHeight());
}

// ========== Pipeline Step 4: Labeling ==========
const btnRunLabeling = document.getElementById('btn-run-labeling');
const toggleExternalDl = document.getElementById('toggle-external-dl');
const externalDlToggleGroup = document.getElementById('external-dl-toggle-group');
const EXTERNAL_DL_ALGOS = ['brits', 'deepmvi', 'mrnn', 'mpin', 'iim'];
const LABELING_ALGO_REFERENCES = {
  STMVL: {
    url: 'https://www.ijcai.org/Proceedings/16/Papers/384.pdf',
    title: 'ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data (IJCAI 2016)',
  },
  CDREC: {
    url: 'https://rdcu.be/b32bv',
    title: 'Scalable Recovery of Missing Blocks in Time Series with High and Low Cross-Correlations (KAIS 2020)',
  },
  SVDIMP: {
    url: 'https://academic.oup.com/bioinformatics/article/17/6/520/272365',
    title: 'Missing value estimation methods for DNA microarrays (Bioinformatics 2001)',
  },
  TRMF: {
    url: 'https://papers.nips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf',
    title: 'Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction (NeurIPS 2016)',
  },
  TKCM: {
    url: 'https://openproceedings.org/2017/conf/edbt/paper-112.pdf',
    title: 'Continuous Imputation of Missing Values in Streams of Pattern-Determining Time Series (EDBT 2017)',
  },
  DYNAMMO: {
    url: 'https://dl.acm.org/doi/10.1145/1557019.1557078',
    title: 'DynaMMo: Mining and Summarization of Coevolving Sequences with Missing Values (KDD 2009)',
  },
  TENMF: {
    url: 'https://proceedings.mlr.press/v70/mei17a/mei17a.pdf',
    title: 'Nonnegative Matrix Factorization for Time Series Recovery From a Few Temporal Aggregates (ICML 2017)',
  },
  SVT: {
    url: 'https://epubs.siam.org/doi/pdf/10.1137/080738970',
    title: 'A Singular Value Thresholding Algorithm for Matrix Completion (SIAM J. Optim. 2010)',
  },
  GROUSE: {
    url: 'https://proceedings.mlr.press/v51/zhang16b.html',
    title: 'Global Convergence of a Grassmannian Gradient Descent Algorithm for Subspace Estimation (AISTATS 2016)',
  },
  SOFTIMP: {
    url: 'https://www.jmlr.org/papers/v11/mazumder10a.html',
    title: 'Spectral Regularization Algorithms for Learning Large Incomplete Matrices (JMLR 2010)',
  },
  ROSL: {
    url: 'https://ieeexplore.ieee.org/document/6909890',
    title: 'Robust Orthonormal Subspace Learning: Efficient Recovery of Corrupted Low-Rank Matrices (CVPR 2014)',
  },
  MRNN: {
    url: 'https://ieeexplore.ieee.org/document/8485748',
    title: 'Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks (IEEE TBME 2019)',
  },
  BRITS: {
    url: 'https://papers.nips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf',
    title: 'BRITS: Bidirectional Recurrent Imputation for Time Series (NeurIPS 2018)',
  },
  DEEPMVI: {
    url: 'https://www.vldb.org/pvldb/vol14/p2533-bansal.pdf',
    title: 'Missing Value Imputation on Multidimensional Time Series (PVLDB 2021)',
  },
  MPIN: {
    url: 'https://www.vldb.org/pvldb/vol17/p345-li.pdf',
    title: 'Missing Value Imputation for Multi-attribute Sensor Data Streams via Message Propagation (PVLDB 2024)',
  },
  IIM: {
    url: 'https://ieeexplore.ieee.org/document/8731351',
    title: 'Learning Individual Models for Imputation (ICDE 2019)',
  },
  PRISTI: {
    url: 'https://ieeexplore.ieee.org/document/10184808',
    title: 'PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation (ICDE 2023)',
  },
};

function initLabelingAlgorithmReferences() {
  const container = document.getElementById('labeling-algorithms');
  if (!container) return;

  container.querySelectorAll('.checkbox-item').forEach((item) => {
    const input = item.querySelector('input[type="checkbox"]');
    const labelSpan = item.querySelector('span');
    if (!input || !labelSpan) return;
    if (labelSpan.dataset.refBound === '1') return;

    const algoKey = String(input.value || '').trim().toUpperCase();
    const ref = LABELING_ALGO_REFERENCES[algoKey];
    if (!ref) return;

    labelSpan.classList.add('algo-ref-label');
    labelSpan.dataset.refUrl = ref.url;
    labelSpan.dataset.refTitle = ref.title;
    labelSpan.title = `${ref.title}\n${ref.url}`;
    labelSpan.setAttribute('tabindex', '0');
    labelSpan.setAttribute('role', 'link');

    const openRef = (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      window.open(ref.url, '_blank', 'noopener,noreferrer');
    };

    labelSpan.addEventListener('click', openRef);
    labelSpan.addEventListener('keydown', (evt) => {
      if (evt.key === 'Enter' || evt.key === ' ') {
        openRef(evt);
      }
    });
    labelSpan.dataset.refBound = '1';
  });
}

let labelingCompleted = false;
if (btnRunLabeling) {
  initLabelingAlgorithmReferences();

  const syncExternalDlToggle = () => {
    const selectedAlgorithms = Array.from(
      document.querySelectorAll('#labeling-algorithms input:checked')
    ).map(cb => cb.value);
    const selectedLower = selectedAlgorithms.map(a => String(a).toLowerCase());
    const hasDlSelected = EXTERNAL_DL_ALGOS.some(a => selectedLower.includes(a));

    if (toggleExternalDl) {
      if (hasDlSelected) {
        toggleExternalDl.checked = true;
        toggleExternalDl.disabled = true;
      } else {
        toggleExternalDl.disabled = false;
      }
    }
    if (externalDlToggleGroup) {
      externalDlToggleGroup.style.display = hasDlSelected ? 'none' : '';
    }
  };

  document.querySelectorAll('#labeling-algorithms input').forEach(cb => {
    cb.addEventListener('change', syncExternalDlToggle);
  });
  syncExternalDlToggle();

  btnRunLabeling.addEventListener('click', async () => {
    // If labeling already completed, just go to next step
    if (labelingCompleted) {
      updatePipelineStepper(5);
      return;
    }

    try {
      // Get selected algorithms
      const selectedAlgorithms = Array.from(
        document.querySelectorAll('#labeling-algorithms input:checked')
      ).map(cb => cb.value);

      if (selectedAlgorithms.length === 0) {
        logPipeline('Please select at least one algorithm', 'error');
        return;
      }

      logPipeline(`Running labeling with ${selectedAlgorithms.length} algorithms...`, 'info');
      btnRunLabeling.classList.add('loading');
      btnRunLabeling.textContent = 'Running...';

      const selectedLower = selectedAlgorithms.map(a => String(a).toLowerCase());
      const hasDlSelected = EXTERNAL_DL_ALGOS.some(a => selectedLower.includes(a));
      const wantsExternalDl = hasDlSelected || !!(toggleExternalDl && toggleExternalDl.checked);
      const chosenExternal = EXTERNAL_DL_ALGOS.find(a => selectedLower.includes(a)) || EXTERNAL_DL_ALGOS[0];
      if (wantsExternalDl) {
        logPipeline(`External DL runner probe enabled`);
      }

      // Call backend API
      let responsePayload;
      if (API_CONFIG.useMock) {
        await simulateDelay(2000);
        responsePayload = { labelingResults: MockData.labelingResults };
      } else {
        responsePayload = await apiCall(API_CONFIG.endpoints.runLabeling, 'POST', {
          datasetId: AppState.pipelineDatasetId,
          algorithms: selectedAlgorithms,
          use_external_dl: wantsExternalDl,
          external_algo: chosenExternal,
        });
      }

      const results = responsePayload.labelingResults || [];
      const externalDl = responsePayload.externalDl || null;

      // Store results
      AppState.labelingResults = results;
      AppState.externalDlStatus = externalDl;

      // Render results
      renderLabelingResults(results);

      if (externalDl && externalDl.requested) {
        if (externalDl.status === 'success') {
          const metricMsg = (typeof externalDl.rmse === 'number')
            ? ` RMSE=${externalDl.rmse.toFixed(4)}`
            : '';
          logPipeline(`External DL runner success (${externalDl.algorithm}).${metricMsg}`, 'success');
        } else if (externalDl.status === 'skipped') {
          logPipeline(`External DL runner skipped: ${externalDl.message || 'no message'}`, 'warn');
        } else if (externalDl.status === 'failed') {
          logPipeline(`External DL runner failed: ${externalDl.error || 'unknown error'}`, 'error');
        } else if (!externalDl.enabled) {
          logPipeline('External DL runner is not enabled on backend.', 'warn');
        }
      }

      logPipeline('Labeling completed!', 'success');
      btnRunLabeling.classList.remove('loading');

      // Mark as completed and change button text
      labelingCompleted = true;
      btnRunLabeling.textContent = 'Continue to ModelRace →';

    } catch (error) {
      logPipeline(`Labeling failed: ${error.message}`, 'error');
      btnRunLabeling.classList.remove('loading');
      btnRunLabeling.textContent = 'Run Labeling';
    }
  });
}

function renderLabelingResults(results) {
  document.getElementById('labeling-placeholder').style.display = 'none';
  const container = document.getElementById('labeling-results');
  container.style.display = 'flex';
  container.innerHTML = results.map(r => `
    <div class="result-item">
      <span class="cluster-name">${r.name}</span>
      <span class="cluster-count">${r.count} series</span>
      <span class="cluster-algo">${r.bestAlgo}</span>
    </div>
  `).join('');
}

/*
// ========== Pipeline Step 2: Feature Extraction ==========
const btnRunFeatures = document.getElementById('btn-run-features');
let featuresCompleted = false;
if (btnRunFeatures) {
  btnRunFeatures.addEventListener('click', async () => {
    // If feature extraction already completed, just go to next step
    if (featuresCompleted) {
      updatePipelineStepper(3);
      return;
    }
    
    try {
      // Get selected feature categories
      const selectedFeatures = Array.from(
        document.querySelectorAll('.feature-checkbox-list input:checked')
      ).map(cb => cb.value);
      
      if (selectedFeatures.length === 0) {
        logPipeline('Please select at least one feature category', 'error');
        return;
      }
      
      logPipeline(`Extracting features: ${selectedFeatures.join(', ')}...`, 'info');
      btnRunFeatures.classList.add('loading');
      btnRunFeatures.textContent = 'Extracting...';
      
      // Call backend API
      let results;
      if (API_CONFIG.useMock) {
        await simulateDelay(2000);
        results = MockData.featureResults;
      } else {
        const response = await apiCall(API_CONFIG.endpoints.runFeatures, 'POST', {
          datasetId: AppState.pipelineDatasetId,
          features: selectedFeatures,
        });
        results = response.featureImportance;
      }
      
      // Store results
      AppState.featureResults = results;
      
      // Render results
      renderFeatureResults(results);
      
      logPipeline('Feature extraction completed!', 'success');
      btnRunFeatures.classList.remove('loading');
      
      // Mark as completed and change button text
      featuresCompleted = true;
      btnRunFeatures.textContent = 'Continue to Clustering →';
      
    } catch (error) {
      logPipeline(`Feature extraction failed: ${error.message}`, 'error');
      btnRunFeatures.classList.remove('loading');
      btnRunFeatures.textContent = 'Extract Features';
    }
  });
}
*/
// ========== Pipeline Step 2: Feature Extraction ==========
const btnRunFeatures = document.getElementById('btn-run-features');
let featuresCompleted = false;
if (btnRunFeatures) {
  btnRunFeatures.addEventListener('click', async () => {
    // If feature extraction already completed, just go to next step
    if (featuresCompleted) {
      updatePipelineStepper(3);
      return;
    }

    try {
      logPipeline('Extracting features: catch22, tsfresh, topological...', 'info');
      btnRunFeatures.classList.add('loading');
      btnRunFeatures.textContent = 'Extracting...';

      // Call backend API
      let featurePayload;
      if (API_CONFIG.useMock) {
        await simulateDelay(2000);
        featurePayload = MockData.featureResults;
      } else {
        featurePayload = await apiCall(API_CONFIG.endpoints.runFeatures, 'POST', {
          datasetId: AppState.pipelineDatasetId,
        });
      }

      const results = normalizeFeaturePayload(featurePayload);
      AppState.featureResults = results;

      // Render results
      renderFeatureResults(results);

      logPipeline('Feature extraction completed!', 'success');
      btnRunFeatures.classList.remove('loading');

      // Mark as completed and change button text
      featuresCompleted = true;
      btnRunFeatures.textContent = 'Continue to Clustering →';

    } catch (error) {
      logPipeline(`Feature extraction failed: ${error.message}`, 'error');
      btnRunFeatures.classList.remove('loading');
      btnRunFeatures.textContent = 'Extract Features';
    }
  });
}

function normalizeFeaturePayload(payload) {
  if (Array.isArray(payload)) {
    return {
      featureImportance: payload,
      featurePreview: {},
      previewRows: 0,
      previewCols: 0,
    };
  }
  return {
    featureImportance: Array.isArray(payload?.featureImportance) ? payload.featureImportance : [],
    featurePreview: payload?.featurePreview && typeof payload.featurePreview === 'object' ? payload.featurePreview : {},
    previewRows: Number.isFinite(Number(payload?.previewRows)) ? Number(payload.previewRows) : 0,
    previewCols: Number.isFinite(Number(payload?.previewCols)) ? Number(payload.previewCols) : 0,
  };
}

let featurePreviewChart = null;

function destroyFeaturePreviewChart() {
  if (featurePreviewChart) {
    featurePreviewChart.destroy();
    featurePreviewChart = null;
  }
}

function toNumericFeatureValue(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function formatFeaturePreviewValue(value) {
  const numeric = toNumericFeatureValue(value);
  if (numeric === null) return '';
  const abs = Math.abs(numeric);
  if (abs >= 1000) return numeric.toFixed(0);
  if (abs >= 100) return numeric.toFixed(1).replace(/\.0$/, '');
  if (abs >= 1) return numeric.toFixed(3).replace(/\.?0+$/, '');
  return numeric.toFixed(4).replace(/\.?0+$/, '');
}

const featurePreviewValueLabelPlugin = {
  id: 'featurePreviewValueLabelPlugin',
  afterDatasetsDraw(chart, _args, pluginOptions) {
    if (pluginOptions?.display === false) return;
    const { ctx, chartArea } = chart;
    const dataset = chart.data?.datasets?.[0];
    const meta = chart.getDatasetMeta(0);
    if (!dataset || !meta?.data?.length || !chartArea) return;

    const color = pluginOptions?.color || '#0f172a';
    const fontSize = pluginOptions?.fontSize || 11;
    const offset = pluginOptions?.offset || 8;
    const paddingRight = pluginOptions?.paddingRight || 6;

    ctx.save();
    ctx.fillStyle = color;
    ctx.font = `600 ${fontSize}px sans-serif`;
    ctx.textBaseline = 'middle';

    meta.data.forEach((bar, index) => {
      const rawValue = dataset.data?.[index];
      const label = formatFeaturePreviewValue(rawValue);
      if (!label) return;

      const xPos = typeof bar?.x === 'number' ? bar.x : null;
      const yPos = typeof bar?.y === 'number' ? bar.y : null;
      const basePos = typeof bar?.base === 'number' ? bar.base : null;
      if (xPos === null || yPos === null || basePos === null) return;

      const textWidth = ctx.measureText(label).width;
      const isPositive = xPos >= basePos;
      let drawX = isPositive ? xPos + offset : xPos - offset;
      let align = isPositive ? 'left' : 'right';

      if (isPositive && drawX + textWidth > chartArea.right - paddingRight) {
        drawX = xPos - offset;
        align = 'right';
      } else if (!isPositive && drawX - textWidth < chartArea.left + paddingRight) {
        drawX = xPos + offset;
        align = 'left';
      }

      ctx.textAlign = align;
      ctx.fillText(label, drawX, yPos);
    });

    ctx.restore();
  },
};

function renderFeaturePreviewChart(entry, extractorKey = '') {
  const controls = document.getElementById('feature-preview-controls');
  const datasetSelect = document.getElementById('feature-preview-dataset-select');
  const rowSelect = document.getElementById('feature-preview-row-select');
  const noteEl = document.getElementById('feature-preview-note');
  const chartWrapper = document.getElementById('feature-preview-chart-wrapper');
  const emptyState = document.getElementById('feature-preview-empty');
  const canvas = document.getElementById('feature-preview-chart');
  if (!controls || !datasetSelect || !rowSelect || !noteEl || !chartWrapper || !emptyState || !canvas) return;

  const rows = Array.isArray(entry?.rows) ? entry.rows : [];
  const sampleColumns = Array.isArray(entry?.sampleColumns) ? entry.sampleColumns : [];
  const idColumn = entry?.idColumn || null;

  if (!entry || rows.length === 0 || sampleColumns.length === 0) {
    destroyFeaturePreviewChart();
    controls.style.display = 'none';
    chartWrapper.style.display = 'none';
    emptyState.style.display = 'block';
    emptyState.textContent = 'No feature preview available for this extractor.';
    return;
  }

  rowSelect.innerHTML = '';
  rows.forEach((row, idx) => {
    const labelValue = idColumn && row?.[idColumn] !== undefined && row?.[idColumn] !== null
      ? row[idColumn]
      : `row_${idx + 1}`;
    const option = document.createElement('option');
    option.value = String(idx);
    option.textContent = String(labelValue);
    rowSelect.appendChild(option);
  });

  const colorMap = {
    catch22: '#3b82f6',
    tsfresh: '#10b981',
    topological: '#f59e0b',
  };
  const barColor = colorMap[String(extractorKey).toLowerCase()] || '#6b7280';

  const renderRowChart = (rowIndex) => {
    const idx = Number.isFinite(Number(rowIndex)) ? Number(rowIndex) : 0;
    const selected = rows[Math.min(Math.max(idx, 0), rows.length - 1)] || {};
    const labels = [];
    const values = [];
    sampleColumns.forEach((col) => {
      const numeric = toNumericFeatureValue(selected[col]);
      if (numeric !== null) {
        labels.push(col);
        values.push(numeric);
      }
    });

    if (values.length === 0) {
      destroyFeaturePreviewChart();
      chartWrapper.style.display = 'none';
      emptyState.style.display = 'block';
      emptyState.textContent = 'Current row has no numeric feature values to visualize.';
      return;
    }

    destroyFeaturePreviewChart();
    const ctx = canvas.getContext('2d');
    featurePreviewChart = new Chart(ctx, {
      type: 'bar',
      plugins: [featurePreviewValueLabelPlugin],
      data: {
        labels,
        datasets: [{
          label: 'Feature Value',
          data: values,
          backgroundColor: barColor,
          borderRadius: 4,
          maxBarThickness: 18,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        layout: {
          padding: { right: 56, left: 12 },
        },
        plugins: {
          legend: { display: false },
          featurePreviewValueLabelPlugin: {
            display: true,
          },
          tooltip: {
            callbacks: {
              title(items) {
                return items?.[0]?.label || '';
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#64748b' },
            grid: { color: '#e2e8f0' },
          },
          y: {
            ticks: {
              color: '#334155',
              callback(value) {
                const label = this.getLabelForValue(value);
                return label.length > 36 ? `${label.slice(0, 33)}...` : label;
              },
            },
            grid: { display: false },
          },
        },
      },
    });

    const fullCount = entry.totalFeatures || sampleColumns.length;
    const rowLabel = idColumn && selected?.[idColumn] !== undefined && selected?.[idColumn] !== null
      ? `${selected[idColumn]}`
      : `row_${idx + 1}`;
    const truncatedMsg = entry.truncated
      ? `Showing sampled ${sampleColumns.length}/${fullCount} features`
      : `Showing ${sampleColumns.length} features`;
    noteEl.textContent = `${truncatedMsg} for time series ${rowLabel}.`;
    chartWrapper.style.display = 'block';
    emptyState.style.display = 'none';
  };

  rowSelect.onchange = (event) => {
    renderRowChart(event.target.value);
  };
  controls.style.display = 'flex';
  renderRowChart(0);
  emptyState.style.display = 'none';
}

function renderFeatureResults(featurePayload) {
  const payload = normalizeFeaturePayload(featurePayload);
  const results = payload.featureImportance;
  const previewRaw = payload.featurePreview;
  const previewByKey = {};
  Object.entries(previewRaw).forEach(([name, entry]) => {
    previewByKey[String(name).toLowerCase()] = entry;
  });

  const placeholder = document.getElementById('features-placeholder');
  const container = document.getElementById('feature-results-container');
  const totalFeaturesEl = document.getElementById('total-features');
  const cardsContainer = document.getElementById('extractor-cards');
  const previewTabs = document.getElementById('feature-preview-tabs');
  const previewDataset = document.getElementById('feature-preview-dataset');
  if (!container || !cardsContainer || !previewTabs || !previewDataset || !totalFeaturesEl) {
    console.warn('[Feature UI] Missing DOM nodes for feature results rendering.');
    return;
  }
  if (placeholder) placeholder.style.display = 'none';
  container.style.display = 'block';

  // Define extractor metadata
  const extractorMeta = {
    catch22: { desc: 'Lightweight canonical time series features' },
    tsfresh: { desc: 'Comprehensive statistical features library' },
    topological: { desc: 'Persistence diagrams from TDA' },
  };

  let cardRows = results.map((item) => {
    const key = String(item?.name || '').toLowerCase();
    const previewEntry = previewByKey[key];
    const previewCount = Number(previewEntry?.totalFeatures);
    const fallbackCount = Number(item?.value);
    const count = Number.isFinite(previewCount) ? previewCount : (Number.isFinite(fallbackCount) ? fallbackCount : 0);
    return {
      key,
      name: item?.name || key,
      value: count,
      datasetsProcessed: Number.isFinite(Number(item?.datasetsProcessed)) ? Number(item.datasetsProcessed) : 0,
      preview: previewEntry || null,
    };
  });

  if (cardRows.length === 0) {
    cardRows = Object.entries(previewByKey).map(([key, previewEntry]) => ({
      key,
      name: key,
      value: Number(previewEntry?.totalFeatures) || 0,
      datasetsProcessed: 0,
      preview: previewEntry,
    }));
  }

  const total = cardRows.reduce((sum, f) => sum + (Number.isFinite(f.value) ? f.value : 0), 0);
  totalFeaturesEl.textContent = total;
  const summaryDataset = document.getElementById('feature-summary-dataset');
  if (summaryDataset) summaryDataset.textContent = '';

  // Render extractor cards
  cardsContainer.innerHTML = cardRows.map((f) => {
    const meta = extractorMeta[f.key] || { desc: 'Feature extractor' };
    const percentage = total > 0 ? ((f.value / total) * 100).toFixed(1) : 0;
    const datasetCountText = f.datasetsProcessed > 0 ? `${f.datasetsProcessed} dataset${f.datasetsProcessed === 1 ? '' : 's'} processed` : 'Dataset count unavailable';
    return `
      <div class="extractor-card ${f.key}">
        <div class="extractor-header">
          <span class="extractor-name">${f.name}</span>
          <span class="extractor-status">✓ Done</span>
        </div>
        <div class="extractor-count">${f.value}</div>
        <div class="extractor-desc">${meta.desc}</div>
        <div class="extractor-percentage">${percentage}% of total · ${datasetCountText}</div>
      </div>
    `;
  }).join('');

  const previewCandidates = cardRows.filter((row) => row.preview);
  const datasetSelect = document.getElementById('feature-preview-dataset-select');

  previewTabs.innerHTML = '';
  if (!datasetSelect || previewCandidates.length === 0) {
    previewDataset.textContent = '';
    if (summaryDataset) summaryDataset.textContent = '';
    renderFeaturePreviewChart(null);
    return;
  }

  const getPreviewEntries = (row) => {
    if (!row?.preview) return [];
    if (Array.isArray(row.preview.datasets) && row.preview.datasets.length > 0) {
      return row.preview.datasets.filter((entry) => entry && entry.dataset);
    }
    return row.preview.dataset ? [row.preview] : [];
  };

  const updateDatasetOptions = (selectedRow, preferredDataset = '') => {
    const entries = getPreviewEntries(selectedRow);
    datasetSelect.innerHTML = '';
    entries.forEach((entry, idx) => {
      const option = document.createElement('option');
      option.value = entry.dataset;
      option.textContent = entry.dataset;
      if ((preferredDataset && entry.dataset === preferredDataset) || (!preferredDataset && idx === 0)) {
        option.selected = true;
      }
      datasetSelect.appendChild(option);
    });
    return entries;
  };

  const selectPreview = (selectedKey, preferredDataset = '') => {
    Array.from(previewTabs.querySelectorAll('.feature-preview-tab')).forEach((tab) => {
      tab.classList.toggle('active', tab.dataset.key === selectedKey);
    });
    const selected = previewCandidates.find((item) => item.key === selectedKey) || previewCandidates[0];
    const datasetEntries = updateDatasetOptions(selected, preferredDataset);
    const activeDatasetName = datasetSelect.value || preferredDataset;
    const activeEntry = datasetEntries.find((entry) => entry.dataset === activeDatasetName) || datasetEntries[0] || selected?.preview || null;
    const dsText = activeEntry?.dataset ? `Dataset: ${activeEntry.dataset}` : '';
    previewDataset.textContent = dsText;
    if (summaryDataset) {
      summaryDataset.textContent = activeEntry?.dataset ? `Preview Dataset: ${activeEntry.dataset}` : '';
    }
    renderFeaturePreviewChart(activeEntry, selected?.key || '');
  };

  datasetSelect.onchange = () => {
    const activeTab = previewTabs.querySelector('.feature-preview-tab.active');
    const activeKey = activeTab?.dataset.key || previewCandidates[0]?.key || '';
    selectPreview(activeKey, datasetSelect.value);
  };

  previewCandidates.forEach((item, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `feature-preview-tab${idx === 0 ? ' active' : ''}`;
    btn.dataset.key = item.key;
    btn.textContent = item.name;
    btn.addEventListener('click', () => selectPreview(item.key));
    previewTabs.appendChild(btn);
  });

  selectPreview(previewCandidates[0].key);
}

let recommendFeaturePreviewChart = null;

function destroyRecommendFeaturePreviewChart() {
  if (recommendFeaturePreviewChart) {
    recommendFeaturePreviewChart.destroy();
    recommendFeaturePreviewChart = null;
  }
}

function renderRecommendFeaturePreviewChart(entry, extractorKey = '') {
  const controls = document.getElementById('recommend-feature-preview-controls');
  const rowSelect = document.getElementById('recommend-feature-preview-row-select');
  const noteEl = document.getElementById('recommend-feature-preview-note');
  const chartWrapper = document.getElementById('recommend-feature-preview-chart-wrapper');
  const emptyState = document.getElementById('recommend-feature-preview-empty');
  const canvas = document.getElementById('recommend-feature-preview-chart');
  if (!controls || !rowSelect || !noteEl || !chartWrapper || !emptyState || !canvas) return;

  const rows = Array.isArray(entry?.rows) ? entry.rows : [];
  const sampleColumns = Array.isArray(entry?.sampleColumns) ? entry.sampleColumns : [];
  const idColumn = entry?.idColumn || null;

  if (!entry || rows.length === 0 || sampleColumns.length === 0) {
    destroyRecommendFeaturePreviewChart();
    controls.style.display = 'none';
    chartWrapper.style.display = 'none';
    emptyState.style.display = 'block';
    emptyState.textContent = 'No feature preview available for this extractor.';
    return;
  }

  rowSelect.innerHTML = '';
  rows.forEach((row, idx) => {
    const labelValue = idColumn && row?.[idColumn] !== undefined && row?.[idColumn] !== null
      ? row[idColumn]
      : `row_${idx + 1}`;
    const option = document.createElement('option');
    option.value = String(idx);
    option.textContent = String(labelValue);
    rowSelect.appendChild(option);
  });

  const colorMap = {
    catch22: '#3b82f6',
    tsfresh: '#10b981',
    topological: '#f59e0b',
  };
  const barColor = colorMap[String(extractorKey).toLowerCase()] || '#6b7280';

  const renderRowChart = (rowIndex) => {
    const idx = Number.isFinite(Number(rowIndex)) ? Number(rowIndex) : 0;
    const selected = rows[Math.min(Math.max(idx, 0), rows.length - 1)] || {};
    const labels = [];
    const values = [];
    sampleColumns.forEach((col) => {
      const numeric = toNumericFeatureValue(selected[col]);
      if (numeric !== null) {
        labels.push(col);
        values.push(numeric);
      }
    });

    if (values.length === 0) {
      destroyRecommendFeaturePreviewChart();
      chartWrapper.style.display = 'none';
      emptyState.style.display = 'block';
      emptyState.textContent = 'Current row has no numeric feature values to visualize.';
      return;
    }

    destroyRecommendFeaturePreviewChart();
    const ctx = canvas.getContext('2d');
    recommendFeaturePreviewChart = new Chart(ctx, {
      type: 'bar',
      plugins: [featurePreviewValueLabelPlugin],
      data: {
        labels,
        datasets: [{
          label: 'Feature Value',
          data: values,
          backgroundColor: barColor,
          borderRadius: 4,
          maxBarThickness: 18,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        layout: {
          padding: { right: 56, left: 12 },
        },
        plugins: {
          legend: { display: false },
          featurePreviewValueLabelPlugin: {
            display: true,
          },
          tooltip: {
            callbacks: {
              title(items) {
                return items?.[0]?.label || '';
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#64748b' },
            grid: { color: '#e2e8f0' },
          },
          y: {
            ticks: {
              color: '#334155',
              callback(value) {
                const label = this.getLabelForValue(value);
                return label.length > 36 ? `${label.slice(0, 33)}...` : label;
              },
            },
            grid: { display: false },
          },
        },
      },
    });

    const fullCount = entry.totalFeatures || sampleColumns.length;
    const rowLabel = idColumn && selected?.[idColumn] !== undefined && selected?.[idColumn] !== null
      ? `${selected[idColumn]}`
      : `row_${idx + 1}`;
    const truncatedMsg = entry.truncated
      ? `Showing sampled ${sampleColumns.length}/${fullCount} features`
      : `Showing ${sampleColumns.length} features`;
    noteEl.textContent = `${truncatedMsg} for time series ${rowLabel}.`;
    chartWrapper.style.display = 'block';
    emptyState.style.display = 'none';
  };

  rowSelect.onchange = (event) => {
    renderRowChart(event.target.value);
  };
  controls.style.display = 'flex';
  renderRowChart(0);
  emptyState.style.display = 'none';
}

function renderRecommendFeatureResults(featurePayload) {
  const payload = normalizeFeaturePayload(featurePayload);
  const results = payload.featureImportance;
  const previewRaw = payload.featurePreview;
  const previewByKey = {};
  Object.entries(previewRaw).forEach(([name, entry]) => {
    previewByKey[String(name).toLowerCase()] = entry;
  });

  const placeholder = document.getElementById('recommend-features-placeholder');
  const container = document.getElementById('recommend-feature-results-container');
  const totalFeaturesEl = document.getElementById('recommend-total-features');
  const cardsContainer = document.getElementById('recommend-extractor-cards');
  const previewTabs = document.getElementById('recommend-feature-preview-tabs');
  const previewDataset = document.getElementById('recommend-feature-preview-dataset');
  if (!container || !cardsContainer || !previewTabs || !previewDataset || !totalFeaturesEl) {
    console.warn('[Recommend Feature UI] Missing DOM nodes for feature results rendering.');
    return;
  }
  if (placeholder) placeholder.style.display = 'none';
  container.style.display = 'block';

  const extractorMeta = {
    catch22: { desc: 'Lightweight canonical time series features' },
    tsfresh: { desc: 'Comprehensive statistical features library' },
    topological: { desc: 'Persistence diagrams from TDA' },
  };

  let cardRows = results.map((item) => {
    const key = String(item?.name || '').toLowerCase();
    const previewEntry = previewByKey[key];
    const previewCount = Number(previewEntry?.totalFeatures);
    const fallbackCount = Number(item?.value);
    const count = Number.isFinite(previewCount) ? previewCount : (Number.isFinite(fallbackCount) ? fallbackCount : 0);
    return {
      key,
      name: item?.name || key,
      value: count,
      datasetsProcessed: Number.isFinite(Number(item?.datasetsProcessed)) ? Number(item.datasetsProcessed) : 0,
      preview: previewEntry || null,
    };
  });

  if (cardRows.length === 0) {
    cardRows = Object.entries(previewByKey).map(([key, previewEntry]) => ({
      key,
      name: key,
      value: Number(previewEntry?.totalFeatures) || 0,
      datasetsProcessed: 0,
      preview: previewEntry,
    }));
  }

  const total = cardRows.reduce((sum, f) => sum + (Number.isFinite(f.value) ? f.value : 0), 0);
  totalFeaturesEl.textContent = total;
  const summaryDataset = document.getElementById('recommend-feature-summary-dataset');
  if (summaryDataset) summaryDataset.textContent = '';

  cardsContainer.innerHTML = cardRows.map((f) => {
    const meta = extractorMeta[f.key] || { desc: 'Feature extractor' };
    const percentage = total > 0 ? ((f.value / total) * 100).toFixed(1) : 0;
    return `
      <div class="extractor-card ${f.key}">
        <div class="extractor-header">
          <span class="extractor-name">${f.name}</span>
          <span class="extractor-status">✓ Done</span>
        </div>
        <div class="extractor-count">${f.value}</div>
        <div class="extractor-desc">${meta.desc}</div>
        <div class="extractor-percentage">${percentage}% of total</div>
      </div>
    `;
  }).join('');

  const previewCandidates = cardRows.filter((row) => row.preview);

  previewTabs.innerHTML = '';
  if (previewCandidates.length === 0) {
    previewDataset.textContent = '';
    if (summaryDataset) summaryDataset.textContent = '';
    renderRecommendFeaturePreviewChart(null);
    return;
  }

  const selectPreview = (selectedKey) => {
    Array.from(previewTabs.querySelectorAll('.feature-preview-tab')).forEach((tab) => {
      tab.classList.toggle('active', tab.dataset.key === selectedKey);
    });
    const selected = previewCandidates.find((item) => item.key === selectedKey) || previewCandidates[0];
    const dsText = selected?.preview?.dataset ? `Dataset: ${selected.preview.dataset}` : '';
    previewDataset.textContent = dsText;
    if (summaryDataset) {
      summaryDataset.textContent = selected?.preview?.dataset ? `Current Dataset: ${selected.preview.dataset}` : '';
    }
    renderRecommendFeaturePreviewChart(selected?.preview || null, selected?.key || '');
  };

  previewCandidates.forEach((item, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `feature-preview-tab${idx === 0 ? ' active' : ''}`;
    btn.dataset.key = item.key;
    btn.textContent = item.name;
    btn.addEventListener('click', () => selectPreview(item.key));
    previewTabs.appendChild(btn);
  });

  selectPreview(previewCandidates[0].key);
}

// ========== Pipeline Step 5: ModelRace ==========
const btnRunModelRace = document.getElementById('btn-run-modelrace');
// Handle ModelRace execution and result rendering.
if (btnRunModelRace) {
  btnRunModelRace.addEventListener('click', async () => {
    try {
      const alpha = parseFloat(document.getElementById('input-alpha').value);
      const beta = parseFloat(document.getElementById('input-beta').value);
      const gamma = parseFloat(document.getElementById('input-gamma').value);
      const seedPipelines = parseInt(document.getElementById('input-seed').value);
      const pValue = parseFloat(document.getElementById('input-pvalue').value);

      logPipeline(`Starting ModelRace (α=${alpha}, β=${beta}, γ=${gamma})...`, 'info');
      btnRunModelRace.classList.add('loading');
      btnRunModelRace.textContent = 'Running...';

      // Call backend API
      let results;
      let evolution;

      if (API_CONFIG.useMock) {
        await simulateDelay(3000);
        results = MockData.modelRaceResults;
        evolution = MockData.pipelineEvolution;
      } else {
        const response = await apiCall(API_CONFIG.endpoints.runModelRace, 'POST', {
          datasetId: AppState.pipelineDatasetId,
          alpha,
          beta,
          gamma,
          seedPipelines,
          pValue,
        });
        results = response.pipelineResults || [];
        evolution = response.evolution || [];  // Guard against undefined.
      }

      // Store results
      AppState.modelRaceResults = results;
      AppState.pipelineEvolution = evolution;

      // Render results
      renderModelRaceResults(results);

      // Render evolution only when data is available.
      if (evolution && evolution.length > 0) {
        renderPipelineEvolution(evolution);
      } else {
        // Show the no-data placeholder.
        const placeholder = document.getElementById('evolution-placeholder');
        if (placeholder) {
          placeholder.style.display = 'block';
        }
      }

      const winner = results[0];
      if (winner) {
        logPipeline(`ModelRace completed! Winner: ${winner.name} (F1: ${winner.f1})`, 'success');
      } else {
        logPipeline('ModelRace completed!', 'success');
      }

      btnRunModelRace.classList.remove('loading');
      btnRunModelRace.textContent = 'Training Complete';
      btnRunModelRace.disabled = true;

      // Update dashboard
      loadDashboardData();

    } catch (error) {
      logPipeline(`ModelRace failed: ${error.message}`, 'error');
      btnRunModelRace.classList.remove('loading');
      btnRunModelRace.textContent = 'Start ModelRace';
    }
  });
}

function renderModelRaceResults(results) {
  document.getElementById('modelrace-placeholder').style.display = 'none';
  const container = document.getElementById('modelrace-results');
  container.style.display = 'flex';
  container.innerHTML = results.map((r, i) => `
    <div class="result-item ${i === 0 ? 'winner' : ''}">
      <span class="cluster-name">${r.name}</span>
      <span class="cluster-algo">F1: ${r.f1.toFixed(2)}</span>
    </div>
  `).join('');
}

// Render Pipeline Evolution
function renderPipelineEvolution(evolution) {
  const container = document.getElementById('pipeline-evolution-container');
  if (!container) return;

  // Exit early when evolution data is empty.
  if (!evolution || evolution.length === 0) {
    console.log('No evolution data available');
    // Show placeholder when no evolution data exists.
    const placeholder = document.getElementById('evolution-placeholder');
    if (placeholder) {
      placeholder.style.display = 'block';
      placeholder.textContent = 'Evolution data not available';
    }
    return;
  }

  container.style.display = 'block';
  document.getElementById('evolution-placeholder')?.style.setProperty('display', 'none');

  // Render progress bar chart
  const chartContainer = document.getElementById('evolution-chart');
  if (chartContainer) {
    const maxCandidates = evolution[0].candidates;
    chartContainer.innerHTML = evolution.map(e => `
      <div class="evolution-bar-row">
        <span class="round-label">R${e.round}</span>
        <div class="evolution-bar-track">
          <div class="evolution-bar-fill" style="width: ${(e.candidates / maxCandidates) * 100}%;">
            <span class="bar-text">${e.candidates}</span>
          </div>
        </div>
        <span class="f1-badge">${e.bestF1.toFixed(2)}</span>
      </div>
    `).join('');
  }

  // Render table
  const tbody = container.querySelector('.evolution-table tbody');
  if (tbody) {
    tbody.innerHTML = evolution.map((e, i) => `
      <tr class="${i === evolution.length - 1 ? 'final-round' : ''}">
        <td>${e.round}</td>
        <td>${e.candidates}</td>
        <td class="eliminated">-${e.eliminated}</td>
        <td>${e.bestF1.toFixed(2)}</td>
        <td class="best-pipeline">${e.bestPipeline}</td>
      </tr>
    `).join('');
  }
}

// ========== Recommend File Upload ==========
const recommendUploadZone = document.getElementById('recommend-upload-zone');
const recommendFileInput = document.getElementById('recommend-file-input');
const btnRecommendContinue = document.getElementById('btn-recommend-continue');
const btnRecommendRunFeatures = document.getElementById('btn-recommend-run-features');
let recommendPreviewRequestId = 0;
let recommendFeaturesCompleted = false;
let recommendationCompleted = false;

function resetRecommendFeatureExtractionState() {
  recommendFeaturesCompleted = false;
  recommendationCompleted = false;
  AppState.recommendFeatureResults = null;
  AppState.recommendResults = null;
  AppState.recommendSeriesList = [];
  AppState.selectedRecoveredSeriesIndex = 0;
  AppState.lastImputationResults = null;
  destroyRecommendFeaturePreviewChart();
  refreshRecoveredSeriesSelector();

  const placeholder = document.getElementById('recommend-features-placeholder');
  const container = document.getElementById('recommend-feature-results-container');
  if (placeholder) {
    placeholder.style.display = 'flex';
    placeholder.innerHTML = '<span>Upload data and run feature extraction to see results</span>';
  }
  if (container) container.style.display = 'none';

  if (btnRecommendRunFeatures) {
    btnRecommendRunFeatures.classList.remove('loading');
    btnRecommendRunFeatures.textContent = 'Extract Features';
    const hasInferenceUpload = !!AppState.recommendDatasetId;
    btnRecommendRunFeatures.disabled = !hasInferenceUpload || AppState.evaluationMode !== 'upload';
  }
}

async function uploadRecommendForPreview() {
  if (AppState.evaluationMode === 'test_set') {
    setRecommendPreviewPlaceholder('Test set mode does not require file preview');
    return;
  }
  if (!AppState.recommendFiles.length) {
    setRecommendPreviewPlaceholder('Upload data to see preview');
    return;
  }

  const requestId = ++recommendPreviewRequestId;
  setRecommendPreviewPlaceholder('Uploading file and generating preview...');

  try {
    const uploadResult = await uploadFiles(API_CONFIG.endpoints.uploadInference, AppState.recommendFiles);
    if (requestId !== recommendPreviewRequestId) return;

    AppState.recommendUploadResult = uploadResult || null;
    AppState.recommendDatasetId = uploadResult?.datasetId || null;
    resetRecommendFeatureExtractionState();

    if (uploadResult?.preview) {
      previewRecommendFile(uploadResult.preview);
      loadRecommendSeriesFromLocalFile();
      logRecommend('Preview generated from uploaded file.', 'success');
    } else {
      setRecommendPreviewPlaceholder('Preview unavailable for this file');
      logRecommend('Preview unavailable for uploaded file.', 'warn');
    }
  } catch (error) {
    if (requestId !== recommendPreviewRequestId) return;
    AppState.recommendUploadResult = null;
    AppState.recommendDatasetId = null;
    resetRecommendFeatureExtractionState();
    setRecommendPreviewPlaceholder('Preview failed. Please try uploading again.');
    logRecommend(`Preview failed: ${error.message}`, 'error');
  }
}

if (recommendUploadZone) {
  recommendUploadZone.addEventListener('click', () => recommendFileInput.click());

  recommendFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      AppState.recommendFiles = [e.target.files[0]];
      AppState.recommendUploadResult = null;
      AppState.recommendDatasetId = null;
      resetRecommendFeatureExtractionState();
      recommendUploadZone.classList.add('has-files');
      btnRecommendContinue.disabled = false;
      logRecommend(`File uploaded: ${e.target.files[0].name}`, 'success');
      loadRecommendSeriesFromLocalFile();
      uploadRecommendForPreview();
    }
  });

  recommendUploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    recommendUploadZone.classList.add('dragover');
  });

  recommendUploadZone.addEventListener('dragleave', () => {
    recommendUploadZone.classList.remove('dragover');
  });

  recommendUploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    recommendUploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      AppState.recommendFiles = [e.dataTransfer.files[0]];
      AppState.recommendUploadResult = null;
      AppState.recommendDatasetId = null;
      resetRecommendFeatureExtractionState();
      recommendUploadZone.classList.add('has-files');
      btnRecommendContinue.disabled = false;
      logRecommend(`File uploaded: ${e.dataTransfer.files[0].name}`, 'success');
      loadRecommendSeriesFromLocalFile();
      uploadRecommendForPreview();
    }
  });
}

if (btnRecommendRunFeatures) {
  btnRecommendRunFeatures.addEventListener('click', async () => {
    if (recommendationCompleted) {
      updateRecommendStepper(3);
      return;
    }

    if (AppState.evaluationMode !== 'upload') {
      try {
        btnRecommendRunFeatures.classList.add('loading');
        btnRecommendRunFeatures.textContent = 'Running...';
        logRecommend('Running recommendation...', 'info');
        await executeRecommendation();
        recommendationCompleted = true;
        btnRecommendRunFeatures.classList.remove('loading');
        btnRecommendRunFeatures.textContent = 'Done';
        updateRecommendStepper(3);
        logRecommend('Recommendation complete!', 'success');
      } catch (error) {
        btnRecommendRunFeatures.classList.remove('loading');
        btnRecommendRunFeatures.textContent = 'Extract Features';
        logRecommend(`Recommendation failed: ${error.message}`, 'error');
      }
      return;
    }
    if (!AppState.recommendDatasetId) {
      logRecommend('Please upload inference data first.', 'error');
      return;
    }

    try {
      btnRecommendRunFeatures.classList.add('loading');
      btnRecommendRunFeatures.textContent = 'Extracting...';
      logRecommend('Extracting features from uploaded incomplete data...', 'info');

      let featurePayload;
      if (API_CONFIG.useMock) {
        await simulateDelay(1200);
        featurePayload = MockData.featureResults;
      } else {
        featurePayload = await apiCall(API_CONFIG.endpoints.runRecommendFeatures, 'POST', {
          datasetId: AppState.recommendDatasetId,
        });
      }

      const normalized = normalizeFeaturePayload(featurePayload);
      AppState.recommendFeatureResults = normalized;
      renderRecommendFeatureResults(normalized);
      recommendFeaturesCompleted = true;

      logRecommend('Feature extraction completed. Running recommendation...', 'info');
      btnRecommendRunFeatures.textContent = 'Recommending...';
      await executeRecommendation();
      recommendationCompleted = true;

      btnRecommendRunFeatures.classList.remove('loading');
      btnRecommendRunFeatures.textContent = 'Done';
      updateRecommendStepper(3);
      logRecommend('Recommendation complete!', 'success');
    } catch (error) {
      btnRecommendRunFeatures.classList.remove('loading');
      btnRecommendRunFeatures.textContent = 'Extract Features';
      logRecommend(`Feature extraction/recommendation failed: ${error.message}`, 'error');
    }
  });
}

async function executeRecommendation() {
  // Handle different evaluation modes
  let uploadResult;
  let recommendPreview = null;

  if (AppState.evaluationMode === 'test_set') {
    logRecommend('Setting up evaluation using 35% test set...', 'info');

    if (API_CONFIG.useMock) {
      await simulateDelay(1500);
      uploadResult = {
        datasetId: 'test-set-dataset',
        n_samples: 50,
        ground_truth_available: true,
      };
    } else {
      uploadResult = await apiCall(API_CONFIG.endpoints.setupTestSet, 'POST', {
        missing_rate: AppState.missingRate,
        missing_pattern: AppState.missingPattern,
      });
    }

    AppState.groundTruthAvailable = true;
    const evalStatus = document.getElementById('evaluation-status');
    const evalStatusMessage = document.getElementById('eval-status-message');
    if (evalStatus) {
      evalStatus.style.display = 'flex';
      if (evalStatusMessage) {
        evalStatusMessage.textContent =
          `Test set ready: ${uploadResult.n_samples || 'N/A'} time series with ${(AppState.missingRate * 100)}% ${AppState.missingPattern} missing`;
      }
    }
  } else if (AppState.evaluationMode === 'complete_upload') {
    logRecommend('Uploading complete data for evaluation...', 'info');

    if (API_CONFIG.useMock) {
      await simulateDelay(1500);
      uploadResult = {
        datasetId: 'complete-upload-dataset',
        ground_truth_available: true,
      };
    } else {
      const fileUploadResult = AppState.recommendUploadResult?.datasetId
        ? AppState.recommendUploadResult
        : await uploadFiles(API_CONFIG.endpoints.uploadInference, AppState.recommendFiles);
      AppState.recommendUploadResult = fileUploadResult;
      recommendPreview = fileUploadResult?.preview || null;

      uploadResult = await apiCall(API_CONFIG.endpoints.setupUpload, 'POST', {
        datasetId: fileUploadResult.datasetId,
        missing_rate: AppState.missingRate,
        missing_pattern: AppState.missingPattern,
      });
    }

    AppState.groundTruthAvailable = true;
  } else {
    if (AppState.recommendUploadResult?.datasetId) {
      uploadResult = AppState.recommendUploadResult;
      recommendPreview = uploadResult?.preview || null;
      logRecommend('Using previously uploaded file and preview.', 'info');
    } else {
      uploadResult = await uploadFiles(API_CONFIG.endpoints.uploadInference, AppState.recommendFiles);
      recommendPreview = uploadResult?.preview || null;
    }
    AppState.groundTruthAvailable = false;
  }

  AppState.recommendDatasetId = uploadResult?.datasetId || 'mock-id';
  if (recommendPreview) previewRecommendFile(recommendPreview);

  let results;
  if (API_CONFIG.useMock) {
    await simulateDelay(1500);
    results = MockData.recommendResults;
  } else {
    results = await apiCall(API_CONFIG.endpoints.getRecommendation, 'POST', {
      datasetId: AppState.recommendDatasetId,
    });
  }

  AppState.recommendResults = results;
  renderRecommendResults(results);
  recommendationCompleted = true;

  if (AppState.groundTruthAvailable) showEvaluationMetricsPanel();

  if (results.ImputePilot) {
    const ImputePilotAlgo = document.getElementById('ImputePilot-algo');
    const ImputePilotConfidence = document.getElementById('ImputePilot-confidence');
    const ImputePilotTime = document.getElementById('ImputePilot-time');
    if (ImputePilotAlgo) ImputePilotAlgo.textContent = results.ImputePilot.algo || '--';
    if (ImputePilotConfidence) {
      ImputePilotConfidence.textContent = results.ImputePilot.confidence
        ? `${(results.ImputePilot.confidence * 100).toFixed(1)}%`
        : '--';
    }
    if (ImputePilotTime) {
      ImputePilotTime.textContent = results.ImputePilot.inference_time_ms
        ? `${results.ImputePilot.inference_time_ms.toFixed(2)} ms`
        : '--';
    }
  }
}

// ========== Recommend Step 1: Continue ==========
if (btnRecommendContinue) {
  btnRecommendContinue.addEventListener('click', async () => {
    try {
      if (AppState.evaluationMode === 'upload' && !AppState.recommendDatasetId) {
        if (!AppState.recommendFiles.length) {
          logRecommend('Please upload inference data first.', 'error');
          return;
        }
        logRecommend('Waiting for upload preview, performing fallback upload...', 'info');
        const uploadResult = await uploadFiles(API_CONFIG.endpoints.uploadInference, AppState.recommendFiles);
        AppState.recommendUploadResult = uploadResult || null;
        AppState.recommendDatasetId = uploadResult?.datasetId || null;
        if (uploadResult?.preview) previewRecommendFile(uploadResult.preview);
        resetRecommendFeatureExtractionState();
      }

      logRecommend('Upload step complete. Moving to feature extraction.', 'success');
      btnRecommendContinue.classList.add('loading');
      btnRecommendContinue.textContent = 'Continuing...';
      if (AppState.evaluationMode === 'upload') {
        updateRecommendStepper(2);
      } else {
        // non-upload modes run recommendation directly
        await executeRecommendation();
        recommendationCompleted = true;
        updateRecommendStepper(3);
        logRecommend('Recommendation complete!', 'success');
      }

      btnRecommendContinue.classList.remove('loading');
      btnRecommendContinue.textContent = 'Continue to Feature Extraction →';
    } catch (error) {
      logRecommend(`Continue failed: ${error.message}`, 'error');
      btnRecommendContinue.classList.remove('loading');
      btnRecommendContinue.textContent = 'Continue to Feature Extraction →';
    }
  });
}

function renderRecommendResults(results) {
  // Render ranking list
  const rankList = document.getElementById('recommend-rank-list');
  if (rankList) {
    const ranking = Array.isArray(results?.ranking) ? results.ranking : [];
    rankList.innerHTML = ranking.length > 0
      ? ranking.map((r, i) => `
        <div class="rank-item ${i === 0 ? 'rank-1' : ''}">
          <span class="rank-num">${r.rank}</span>
          <span class="rank-name">${r.algo}</span>
        </div>
      `).join('')
      : `<div class="rank-item"><span class="rank-num">-</span><span class="rank-name">No ranking available</span></div>`;
  }

  // Render voting matrix
  const votingTable = document.getElementById('voting-table');
  if (votingTable) {
    const rows = Array.isArray(results?.votingMatrix) ? results.votingMatrix : [];
    const thead = votingTable.querySelector('thead');
    const tbody = votingTable.querySelector('tbody');

    const fallbackColumns = rows.reduce((max, row) => {
      if (Array.isArray(row?.pipelineScores)) return Math.max(max, row.pipelineScores.length);
      return Math.max(max, 3);
    }, 0);
    const pipelineHeaders = Array.isArray(results?.pipelineHeaders) && results.pipelineHeaders.length > 0
      ? results.pipelineHeaders
      : Array.from({ length: fallbackColumns || 3 }, (_, i) => `P${i + 1}`);
    const pipelineCount = pipelineHeaders.length;
    const totalCols = pipelineCount + 2;

    if (thead) {
      thead.innerHTML = `
        <tr>
          <th>Algorithm</th>
          ${pipelineHeaders.map((header) => `<th>${header}</th>`).join('')}
          <th>Average</th>
        </tr>
      `;
    }

    if (tbody) {
      const formatScore = (value) => {
        const num = Number(value);
        return Number.isFinite(num) ? num.toFixed(2) : '--';
      };
      tbody.innerHTML = rows.length > 0
        ? rows.map((row) => {
          const scores = Array.isArray(row?.pipelineScores)
            ? row.pipelineScores
            : [row?.p1, row?.p2, row?.p3];
          const normalizedScores = pipelineHeaders.map((_, index) => scores[index]);
          const avg = Number(row?.avg);
          return `
            <tr>
              <td>${row?.algo || '--'}</td>
              ${normalizedScores.map((score) => `<td>${formatScore(score)}</td>`).join('')}
              <td><strong>${Number.isFinite(avg) ? avg.toFixed(2) : '--'}</strong></td>
            </tr>
          `;
        }).join('')
        : `<tr><td colspan="${totalCols}">No voting data available</td></tr>`;
    }
  }

  const pipelinesMeta = document.getElementById('voting-pipelines-meta');
  if (pipelinesMeta) {
    const used = Number(results?.pipelinesUsed);
    const configured = Number(results?.pipelinesConfigured);
    if (Number.isFinite(used) && Number.isFinite(configured)) {
      pipelinesMeta.textContent = `Pipelines used: ${used}/${configured}. Columns reflect all active pipelines; Average is computed over active pipelines.`;
    } else if (Number.isFinite(used)) {
      pipelinesMeta.textContent = `Pipelines used: ${used}. Columns reflect all active pipelines.`;
    } else {
      pipelinesMeta.textContent = '';
    }
  }

  // Render baseline cards (multi-select version)
  renderBaselineCards();
}

// ========== Recommend Baseline Tabs (Multi-select) ==========
document.querySelectorAll('.baseline-tab input[type="checkbox"]').forEach(checkbox => {
  checkbox.addEventListener('change', function () {
    const baselineName = this.value;

    if (this.checked) {
      // Add to selected list
      if (!AppState.selectedBaselines.includes(baselineName)) {
        AppState.selectedBaselines.push(baselineName);
      }
    } else {
      // Remove from selected list
      AppState.selectedBaselines = AppState.selectedBaselines.filter(b => b !== baselineName);
    }

    // Re-render all selected baseline cards
    renderBaselineCards();
  });
});

function renderBaselineCards() {
  const container = document.getElementById('baseline-details');
  if (!container) return;

  // ImputePilot card is always displayed
  const ImputePilotData = AppState.recommendResults?.ImputePilot;
  let html = `
    <div class="baseline-card ImputePilot-card">
      <h4>${BRAND_NAME}</h4>
      <div class="baseline-info">
        <div>Recommend Algorithm: <span>${ImputePilotData?.algo || '--'}</span></div>
        <div>Confidence: <span>${ImputePilotData?.confidence?.toFixed(4) || '--'}</span></div>
        <div>Inference Time: <span>${ImputePilotData?.inference_time_ms?.toFixed(2) || '--'} ms</span></div>
      </div>
    </div>
  `;

  // Generate card for each selected baseline
  AppState.selectedBaselines.forEach(baselineName => {
    let baselineData;
    if (API_CONFIG.useMock) {
      baselineData = MockData.baselineData[baselineName];
    } else {
      baselineData = AppState.recommendResults?.baselines?.[baselineName];
    }

    if (baselineData) {
      // Check if baseline is trained
      if (baselineData.trained === false) {
        const isTraining = AppState.baselineTraining?.[baselineName] === true;
        html += `
          <div class="baseline-card baseline-not-trained">
            <h4>${baselineName}</h4>
            <div class="baseline-info">
              <div class="not-trained-message">${isTraining ? 'Training in progress...' : (baselineData.message || 'Not trained')}</div>
            </div>
          </div>
        `;
      } else if (baselineData.error) {
        html += `
          <div class="baseline-card baseline-error">
            <h4>${baselineName}</h4>
            <div class="baseline-info">
              <div class="error-message">Error: ${baselineData.error}</div>
            </div>
          </div>
        `;
      } else {
        html += `
          <div class="baseline-card">
            <h4>${baselineName} <span class="trained-badge">✓ Trained</span></h4>
            <div class="baseline-info">
              <div>Recommend Algorithm: <span>${baselineData.algo || '--'}</span></div>
              <div>Confidence: <span>${baselineData.confidence?.toFixed(4) || '--'}</span></div>
              <div>Inference Time: <span>${baselineData.inference_time_ms?.toFixed(2) || '--'} ms</span></div>
              <div>Training F1: <span>${baselineData.f1_train?.toFixed(4) || '--'}</span></div>
              <div>Best Estimator: <span>${baselineData.best_estimator || '--'}</span></div>
            </div>
          </div>
        `;
      }
    }
  });

  container.innerHTML = html;

}

// ========== Baseline Training Functions ==========
async function trainBaseline(baselineName) {
  // Endpoint mapping for each baseline
  const endpointMap = {
    'FLAML': API_CONFIG.endpoints.trainFlamlBaseline,
    'Tune': API_CONFIG.endpoints.trainTuneBaseline,
    'AutoFolio': API_CONFIG.endpoints.trainAutofolioBaseline,
    'RAHA': API_CONFIG.endpoints.trainRahaBaseline,
  };

  const endpoint = endpointMap[baselineName];
  if (!endpoint) {
    logRecommend(`${baselineName} training is not implemented yet.`, 'warn');
    return;
  }

  AppState.baselineTraining[baselineName] = true;
  renderBaselineCards();
  logRecommend(`Starting ${baselineName} training...`, 'info');

  try {
    // Build request body based on baseline type
    let requestBody = { time_budget: 3600 };  // 1 hour default

    if (baselineName === 'Tune') {
      // Tune-specific parameters
      requestBody = {
        classifier: 'RandomForest',  // Default classifier (paper: single classifier at a time)
        time_budget: 300,
        num_samples: 50,  // Pre-generated configurations
      };
    }

    const response = await apiCall(endpoint, 'POST', requestBody);

    if (response.status === 'success') {
      let successMsg = `${baselineName} training complete! `;
      if (baselineName === 'Tune') {
        successMsg += `Classifier: ${response.classifier}, F1: ${response.f1_score}, Trials: ${response.num_trials}`;
      } else {
        successMsg += `Best estimator: ${response.best_estimator}, F1: ${response.f1_score}`;
      }
      logRecommend(successMsg, 'success');

      // Update baseline status
      AppState.baselineStatus[baselineName] = {
        trained: true,
        ...response,
      };

      // Re-fetch recommendation to get updated baseline predictions
      await refreshRecommendation();
    } else {
      logRecommend(`${baselineName} training failed: ${response.error}`, 'error');
    }
  } catch (error) {
    logRecommend(`${baselineName} training failed: ${error.message}`, 'error');
  } finally {
    AppState.baselineTraining[baselineName] = false;
    renderBaselineCards();
  }
}

async function refreshRecommendation() {
  logRecommend('Refreshing recommendation with trained baselines...', 'info');

  try {
    const response = await apiCall(API_CONFIG.endpoints.getRecommendation, 'POST', {
      datasetId: AppState.recommendDatasetId,
    });

    AppState.recommendResults = response;
    renderRecommendResults(response);
    logRecommend('Recommendation refreshed!', 'success');
  } catch (error) {
    logRecommend(`Failed to refresh recommendation: ${error.message}`, 'error');
  }
}

async function runBaselineComparison() {
  logRecommend('Running baseline comparison (this may take a while)...', 'info');

  try {
    const response = await apiCall(API_CONFIG.endpoints.compareBaselines, 'POST', {
      baselines: AppState.selectedBaselines,
    });

    if (response.results) {
      renderComparisonTable(response.results);
      logRecommend(`Comparison complete! Best method: ${response.summary?.best_method}`, 'success');
    }
  } catch (error) {
    logRecommend(`Baseline comparison failed: ${error.message}`, 'error');
  }
}

function renderComparisonTable(results) {
  const comparisonTable = document.getElementById('comparison-table');
  if (!comparisonTable) return;

  const tbody = comparisonTable.querySelector('tbody');
  if (!tbody) return;

  tbody.innerHTML = results.map((r, i) => {
    const isAdarts = PRIMARY_METHOD_ALIASES.has(r.method);
    const improvement = r.rmse_improvement_vs_ImputePilot;
    let improvementText = '—';
    let improvementClass = '';

    if (improvement !== undefined && improvement !== null) {
      if (improvement > 0) {
        improvementText = `+${improvement.toFixed(1)}% better`;
        improvementClass = 'positive';
      } else if (improvement < 0) {
        improvementText = `${improvement.toFixed(1)}% worse`;
        improvementClass = 'negative';
      }
    }

    return `
      <tr class="${isAdarts ? 'highlight' : ''}">
        <td>${displayMethodName(r.method)}</td>
        <td>${r.algorithm || '--'}</td>
        <td>${r.rmse !== null ? r.rmse.toFixed(4) : '--'}</td>
        <td>${r.mae !== null ? r.mae.toFixed(4) : '--'}</td>
        <td>${r.runtime_seconds !== null ? r.runtime_seconds.toFixed(2) + 's' : '--'}</td>
        <td class="${improvementClass}">${improvementText}</td>
      </tr>
    `;
  }).join('');
}

// ========== Recommend Step 4: Get Imputation ==========
const btnGetImputation = document.getElementById('btn-get-imputation');
if (btnGetImputation) {
  btnGetImputation.addEventListener('click', async () => {
    if (!recommendationCompleted || !AppState.recommendResults) {
      logRecommend('Please run recommendation first.', 'error');
      return;
    }
    try {
      logRecommend('Running imputation and baseline comparison...', 'info');
      btnGetImputation.classList.add('loading');
      btnGetImputation.textContent = 'Processing...';

      // Call backend API for imputation
      let results;
      if (API_CONFIG.useMock) {
        await simulateDelay(2000);
        results = MockData.imputationResults;
      } else {
        const response = await apiCall(API_CONFIG.endpoints.runImputation, 'POST', {
          datasetId: AppState.recommendDatasetId,
          algorithm: AppState.recommendResults?.ranking?.[0]?.algo,
        });
        results = response;
      }

      // Render imputation results
      renderImputationResults(results);

      logRecommend('Imputation complete!', 'success');

      // Run baseline comparison if any baselines are selected and trained
      const trainedBaselines = AppState.selectedBaselines.filter(b => {
        const baselineData = AppState.recommendResults?.baselines?.[b];
        return baselineData && baselineData.trained === true;
      });

      if (trainedBaselines.length > 0) {
        logRecommend('Running baseline comparison...', 'info');
        await runBaselineComparison();
      }

      btnGetImputation.classList.remove('loading');
      btnGetImputation.textContent = 'Get Imputation';
      updateRecommendStepper(4);

    } catch (error) {
      logRecommend(`Imputation failed: ${error.message}`, 'error');
      btnGetImputation.classList.remove('loading');
      btnGetImputation.textContent = 'Get Imputation';
    }
  });
}

function renderImputationResults(results) {
  AppState.lastImputationResults = results;

  // Update metric cards (with null checks)
  const resultAlgo = document.getElementById('result-algo');
  const resultMissing = document.getElementById('result-missing');
  const resultRate = document.getElementById('result-rate');
  const resultTime = document.getElementById('result-time');

  if (resultAlgo) resultAlgo.textContent = results.algo;
  if (resultMissing) resultMissing.textContent = results.missingPoints;
  if (resultRate) resultRate.textContent = results.recoveryRate;
  if (resultTime) resultTime.textContent = results.processingTime;

  // Update comparison table
  const comparisonTable = document.getElementById('comparison-table');
  if (comparisonTable) {
    const tbody = comparisonTable.querySelector('tbody');
    if (results.comparison && tbody) {
      tbody.innerHTML = results.comparison.map((r, i) => `
        <tr class="${i === 0 ? 'highlight' : ''}">
          <td>${displayMethodName(r.method)}</td>
          <td>${r.algo}</td>
          <td>${r.rmse}</td>
          <td>${r.mae}</td>
          <td>${r.runtime}</td>
          <td class="${r.improvement ? 'positive' : ''}">${r.improvement || '—'}</td>
        </tr>
      `).join('');
    } else if (tbody) {
      const rmseVal = results.rmse ?? '--';
      const maeVal = results.mae ?? '--';
      const runtimeVal = results.processingTime ?? results.runtime ?? '--';
      const algoVal = results.algo ?? '--';
      tbody.innerHTML = `
        <tr class="highlight">
          <td>${BRAND_NAME}</td>
          <td>${algoVal}</td>
          <td>${typeof rmseVal === 'number' ? rmseVal.toFixed(4) : rmseVal}</td>
          <td>${typeof maeVal === 'number' ? maeVal.toFixed(4) : maeVal}</td>
          <td>${runtimeVal}</td>
          <td>—</td>
        </tr>
      `;
    }
  }

  if ((!Array.isArray(AppState.recommendSeriesList) || AppState.recommendSeriesList.length === 0) && AppState.recommendUploadResult?.preview) {
    AppState.recommendSeriesList = buildPipelinePreviewSeries(AppState.recommendUploadResult.preview);
  }
  refreshRecoveredSeriesSelector();

  const seriesList = Array.isArray(AppState.recommendSeriesList) ? AppState.recommendSeriesList : [];
  const seriesIdxRaw = Number(AppState.selectedRecoveredSeriesIndex);
  const seriesIdx = Number.isFinite(seriesIdxRaw)
    ? Math.min(Math.max(seriesIdxRaw, 0), Math.max(seriesList.length - 1, 0))
    : 0;
  const selectedSeries = seriesList.length ? seriesList[seriesIdx] : null;
  const baseChartData = selectedSeries?.chartData || results?.imputedChartData || results?.chartData || AppState.recommendUploadResult?.preview?.chartData || [];
  let imputedValues = null;
  if (Array.isArray(results?.imputedSeries)) {
    imputedValues = results.imputedSeries.map(v => (typeof v === 'object' ? v.y : v));
  } else if (Array.isArray(results?.imputedValues)) {
    imputedValues = results.imputedValues;
  } else if (baseChartData && baseChartData.some(p => p && p.imputed !== undefined)) {
    imputedValues = baseChartData.map(p => (p ? p.imputed : null));
  }
  renderImputedTimeSeriesChart(baseChartData, imputedValues);
}

const recoveredSeriesSelect = document.getElementById('recovered-series-select');
if (recoveredSeriesSelect) {
  recoveredSeriesSelect.addEventListener('change', (event) => {
    const idx = parseInt(event.target.value, 10);
    AppState.selectedRecoveredSeriesIndex = Number.isFinite(idx) ? idx : 0;
    refreshRecoveredSeriesSelector();
    if (AppState.lastImputationResults) {
      renderImputationResults(AppState.lastImputationResults);
    }
  });
}

// ========== Recommend Step 5: Downstream ==========
const btnDownstream = document.getElementById('btn-downstream');
if (btnDownstream) {
  btnDownstream.addEventListener('click', async () => {
    logRecommend('Preparing downstream evaluation (forecasting + classification)...', 'info');
    btnDownstream.classList.add('loading');
    btnDownstream.textContent = 'Preparing...';
    btnDownstream.disabled = true;

    try {
      const tasks = ['forecasting', 'classification'];
      await Promise.all(tasks.map(task => fetchDownstreamResults(task)));
      updateRecommendStepper(5);

      const activeTaskCard = document.querySelector('.task-card.active');
      const defaultTask = activeTaskCard?.dataset?.task || 'forecasting';
      await loadDownstreamResults(defaultTask);
      logRecommend('Downstream evaluation ready.', 'success');
    } catch (error) {
      logRecommend(`Downstream evaluation failed: ${error.message}`, 'error');
    } finally {
      btnDownstream.classList.remove('loading');
      btnDownstream.textContent = 'Downstream Evaluation →';
      btnDownstream.disabled = false;
    }
  });
}

// Download button
const btnDownload = document.getElementById('btn-download');
if (btnDownload) {
  btnDownload.addEventListener('click', async () => {
    logRecommend('Downloading recovered data...', 'info');

    if (API_CONFIG.useMock) {
      alert('Download functionality will be implemented with backend');
    } else {
      // Trigger file download from backend
      window.location.href = `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.downloadResult}?datasetId=${AppState.recommendDatasetId}`;
    }
  });
}

// ========== Recommend Step 5: Task Selection ==========
document.querySelectorAll('.task-card').forEach(card => {
  card.addEventListener('click', async function () {
    document.querySelectorAll('.task-card').forEach(c => c.classList.remove('active'));
    this.classList.add('active');

    const task = this.dataset.task;
    logRecommend(`Selected task: ${task}`, 'info');

    await loadDownstreamResults(task);
  });
});

async function fetchDownstreamResults(task) {
  const cached = AppState.downstreamResults?.[task];
  if (cached) return cached;

  let results;
  if (API_CONFIG.useMock) {
    results = MockData.downstreamResults[task];
  } else {
    const response = await apiCall(API_CONFIG.endpoints.runDownstream, 'POST', {
      datasetId: AppState.recommendDatasetId,
      task,
    });
    results = response;
  }

  if (!AppState.downstreamResults) {
    AppState.downstreamResults = {};
  }
  AppState.downstreamResults[task] = results;
  return results;
}

function renderDownstreamResults(task, results) {
  if (!results) return;

  // Update metric description
  const metricDesc = document.getElementById('downstream-metric-desc');
  if (metricDesc) {
    metricDesc.textContent = task === 'forecasting'
      ? 'RMSE (lower is better)'
      : 'Accuracy (higher is better)';
  }
  const metricLabel = task === 'forecasting' ? 'RMSE' : 'Accuracy';
  const metricAdartsEl = document.getElementById('downstream-metric-imputepilot')
    || document.getElementById('downstream-metric-ImputePilot');
  if (metricAdartsEl) metricAdartsEl.textContent = metricLabel;
  const metricBaselineEl = document.getElementById('downstream-metric-baseline');
  if (metricBaselineEl) metricBaselineEl.textContent = metricLabel;

  // Support both old and new format
  const gtValue = results.groundTruth?.value ?? results.groundTruth ?? '--';
  const gtStd = results.groundTruth?.std ?? '--';
  const ImputePilotValue = results.withImputePilot?.value ?? results.withImputePilot ?? results.withAdarts?.value ?? results.withAdarts;
  const ImputePilotStd = results.withImputePilot?.std ?? results.withImputePilotStd ?? results.withAdarts?.std ?? results.withAdartsStd ?? '--';
  const baselineValue = results.withoutImputePilot?.value ?? results.withoutImputePilot ?? results.withoutAdarts?.value ?? results.withoutAdarts;
  const baselineStd = results.withoutImputePilot?.std ?? results.withoutImputePilotStd ?? results.withoutAdarts?.std ?? results.withoutAdartsStd ?? '--';

  // Update Ground Truth card
  const gtValueEl = document.getElementById('downstream-gt-value');
  const gtStdEl = document.getElementById('downstream-gt-std');
  if (gtValueEl) gtValueEl.textContent = typeof gtValue === 'number' ? gtValue.toFixed(3) : gtValue;
  if (gtStdEl) gtStdEl.textContent = `± ${typeof gtStd === 'number' ? gtStd.toFixed(3) : gtStd}`;

  // Update ImputePilot card
  const ImputePilotValueEl = document.getElementById('downstream-imputepilot-value')
    || document.getElementById('downstream-ImputePilot-value');
  const ImputePilotStdEl = document.getElementById('downstream-imputepilot-std')
    || document.getElementById('downstream-ImputePilot-std');
  if (ImputePilotValueEl) ImputePilotValueEl.textContent = typeof ImputePilotValue === 'number' ? ImputePilotValue.toFixed(3) : ImputePilotValue;
  if (ImputePilotStdEl) ImputePilotStdEl.textContent = `± ${typeof ImputePilotStd === 'number' ? ImputePilotStd.toFixed(3) : ImputePilotStd}`;

  // Update Baseline card
  const baselineValueEl = document.getElementById('downstream-baseline-value');
  const baselineStdEl = document.getElementById('downstream-baseline-std');
  if (baselineValueEl) baselineValueEl.textContent = typeof baselineValue === 'number' ? baselineValue.toFixed(3) : baselineValue;
  if (baselineStdEl) baselineStdEl.textContent = `± ${typeof baselineStd === 'number' ? baselineStd.toFixed(3) : baselineStd}`;

  const improvementCardEl = document.getElementById('downstream-improvement');
  if (improvementCardEl) {
    const improvementText = results.improvement != null
      ? `+${results.improvement.toFixed ? results.improvement.toFixed(1) : results.improvement}%`
      : '--';
    improvementCardEl.textContent = improvementText;
  }

  // Update Summary section
  const summarySection = document.getElementById('downstream-summary');
  if (summarySection) {
    summarySection.style.display = 'block';

    const improvementEl = document.getElementById('downstream-improvement');
    const gapEl = document.getElementById('downstream-gap');
    const countEl = document.getElementById('downstream-count');

    if (improvementEl) improvementEl.textContent = `+${results.improvement?.toFixed(1) ?? results.improvement ?? '--'}%`;
    if (gapEl) gapEl.textContent = `${results.gapToOptimal?.toFixed(1) ?? results.gapToOptimal ?? '--'}%`;
    if (countEl) countEl.textContent = results.n_evaluated ?? '--';
  }

  // Update details section
  const detailsSection = document.getElementById('downstream-details');
  const detailsBody = document.getElementById('downstream-details-body');
  if (detailsSection && detailsBody) {
    detailsSection.style.display = 'block';
    detailsBody.innerHTML = `
      <tr>
        <td><strong>Ground Truth (Optimal)</strong></td>
        <td>${typeof gtValue === 'number' ? gtValue.toFixed(3) : gtValue}</td>
        <td>± ${typeof gtStd === 'number' ? gtStd.toFixed(3) : gtStd}</td>
        <td>Best possible result using oracle imputation</td>
      </tr>
      <tr>
        <td><strong>With ${BRAND_NAME}</strong></td>
        <td>${typeof ImputePilotValue === 'number' ? ImputePilotValue.toFixed(3) : ImputePilotValue}</td>
        <td>± ${typeof ImputePilotStd === 'number' ? ImputePilotStd.toFixed(3) : ImputePilotStd}</td>
        <td>Using ${BRAND_NAME} recommended algorithm</td>
      </tr>
      <tr>
        <td><strong>Without ${BRAND_NAME}</strong></td>
        <td>${typeof baselineValue === 'number' ? baselineValue.toFixed(3) : baselineValue}</td>
        <td>± ${typeof baselineStd === 'number' ? baselineStd.toFixed(3) : baselineStd}</td>
        <td>Using mean-filled baseline</td>
      </tr>
    `;
  }

  logRecommend(`Loaded ${task} results: Improvement ${results.improvement}% vs baseline`, 'success');
}

async function loadDownstreamResults(task) {
  const cached = AppState.downstreamResults?.[task];
  const results = cached || await fetchDownstreamResults(task);
  renderDownstreamResults(task, results);
}

// ========== Utilities ==========
function formatSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
}

function simulateDelay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ========== Train Selected Baselines Button ==========
const btnTrainSelected = document.getElementById('btn-train-selected');
if (btnTrainSelected) {
  btnTrainSelected.addEventListener('click', async () => {
    const selectedToTrain = AppState.selectedBaselines.filter(b => {
      // Only train if not already trained or in-flight
      const status = AppState.recommendResults?.baselines?.[b] || AppState.baselineStatus?.[b];
      const isTraining = AppState.baselineTraining?.[b] === true;
      return !status?.trained && !isTraining;
    });

    if (selectedToTrain.length === 0) {
      logRecommend('No baselines to train (already trained or not implemented).', 'info');
      return;
    }

    btnTrainSelected.classList.add('loading');
    btnTrainSelected.textContent = 'Training...';
    btnTrainSelected.disabled = true;

    for (const baseline of selectedToTrain) {
      await trainBaseline(baseline);
    }

    btnTrainSelected.classList.remove('loading');
    btnTrainSelected.textContent = 'Train Selected Baselines';
    btnTrainSelected.disabled = false;
  });
}

// ========== Global Exports ==========
window.removePipelineFile = removePipelineFile;
window.updatePipelineStepper = updatePipelineStepper;
window.updateRecommendStepper = updateRecommendStepper;
window.AppState = AppState;
window.API_CONFIG = API_CONFIG;
window.trainBaseline = trainBaseline;
window.runBaselineComparison = runBaselineComparison;

window.addEventListener('resize', () => {
  if (AppState.pipelineStep === 3) {
    syncClusterPanelsHeight();
  }
});

// ========== Evaluation Mode Event Handlers ==========
document.querySelectorAll('input[name="eval-mode"]').forEach(radio => {
  radio.addEventListener('change', function () {
    AppState.evaluationMode = this.value;
    const injectionSettings = document.getElementById('missing-injection-settings');
    const uploadZone = document.getElementById('recommend-upload-zone');
    const uploadText = document.getElementById('upload-text');

    if (this.value === 'test_set' || this.value === 'complete_upload') {
      // Show missing injection settings
      if (injectionSettings) injectionSettings.style.display = 'block';
    } else {
      if (injectionSettings) injectionSettings.style.display = 'none';
    }

    // Update upload zone text based on mode
    if (uploadText) {
      switch (this.value) {
        case 'upload':
          uploadText.textContent = 'Click to upload your incomplete data';
          break;
        case 'test_set':
          uploadText.textContent = 'Using 35% test set from training - No upload needed';
          break;
        case 'complete_upload':
          uploadText.textContent = 'Click to upload complete data for evaluation';
          break;
      }
    }

    // If test_set mode, enable the continue button directly
    if (this.value === 'test_set') {
      document.getElementById('btn-recommend-continue').disabled = false;
      if (btnRecommendRunFeatures) btnRecommendRunFeatures.disabled = true;
      resetRecommendFeatureExtractionState();
      logRecommend('Test set mode selected. Click "Continue" to proceed.', 'info');
    } else if (this.value === 'upload' || this.value === 'complete_upload') {
      // Re-check if files are uploaded
      document.getElementById('btn-recommend-continue').disabled = AppState.recommendFiles.length === 0;
      if (btnRecommendRunFeatures) {
        const hasInferenceUpload = !!AppState.recommendDatasetId;
        btnRecommendRunFeatures.disabled = !hasInferenceUpload || this.value !== 'upload';
      }
      if (this.value !== 'upload') {
        resetRecommendFeatureExtractionState();
      }
    }

    logRecommend(`Evaluation mode changed to: ${this.value}`, 'info');
  });
});

// Missing rate slider
const missingRateSlider = document.getElementById('missing-rate-slider');
const missingRateValue = document.getElementById('missing-rate-value');
if (missingRateSlider && missingRateValue) {
  missingRateSlider.addEventListener('input', function () {
    const value = parseInt(this.value);
    missingRateValue.textContent = `${value}%`;
    AppState.missingRate = value / 100;
  });
}

// Missing pattern select
const missingPatternSelect = document.getElementById('missing-pattern-select');
if (missingPatternSelect) {
  missingPatternSelect.addEventListener('change', function () {
    AppState.missingPattern = this.value;
    logRecommend(`Missing pattern changed to: ${this.value}`, 'info');
  });
}

// ========== Evaluation Metrics Functions ==========
const btnComputeGtLabels = document.getElementById('btn-compute-gt-labels');
if (btnComputeGtLabels) {
  btnComputeGtLabels.addEventListener('click', async () => {
    const metricsStatus = document.getElementById('metrics-status');
    if (metricsStatus) {
      metricsStatus.classList.add('loading');
      metricsStatus.innerHTML = '<span class="spinner"></span> Computing ground truth labels... This may take a while.';
    }

    btnComputeGtLabels.classList.add('loading');
    btnComputeGtLabels.disabled = true;

    try {
      let response;
      if (API_CONFIG.useMock) {
        await simulateDelay(3000);
        response = {
          status: 'success',
          message: 'Ground truth labels computed for 50 time series',
          n_samples: 50,
          label_distribution: { 'ROSL': 15, 'CDRec': 12, 'BRITS': 10, 'SVDImp': 8, 'Other': 5 }
        };
      } else {
        response = await apiCall(API_CONFIG.endpoints.computeGroundTruthLabels, 'POST', {
          use_imputebench: false  // Set to true for real ImputeBench
        });
      }

      AppState.groundTruthLabelsComputed = true;

      if (metricsStatus) {
        metricsStatus.classList.remove('loading');
        metricsStatus.classList.add('success');
        metricsStatus.textContent = `✓ ${response.message || 'Ground truth labels computed successfully'}`;
      }

      // Enable Get Metrics button
      const btnGetMetrics = document.getElementById('btn-get-metrics');
      if (btnGetMetrics) {
        btnGetMetrics.disabled = false;
      }

      logRecommend('Ground truth labels computed. Click "Get Evaluation Metrics" to see results.', 'success');

    } catch (error) {
      if (metricsStatus) {
        metricsStatus.classList.remove('loading');
        metricsStatus.textContent = `Error: ${error.message}`;
      }
      logRecommend(`Error computing ground truth labels: ${error.message}`, 'error');
    } finally {
      btnComputeGtLabels.classList.remove('loading');
      btnComputeGtLabels.disabled = false;
    }
  });
}

const btnGetMetrics = document.getElementById('btn-get-metrics');
if (btnGetMetrics) {
  btnGetMetrics.addEventListener('click', async () => {
    const metricsStatus = document.getElementById('metrics-status');
    if (metricsStatus) {
      metricsStatus.classList.add('loading');
      metricsStatus.innerHTML = '<span class="spinner"></span> Calculating evaluation metrics...';
    }

    btnGetMetrics.classList.add('loading');

    try {
      let response;
      if (API_CONFIG.useMock) {
        await simulateDelay(1500);
        response = MockData.evaluationMetrics;
      } else {
        response = await apiCall(API_CONFIG.endpoints.evaluationMetrics, 'GET');
      }

      AppState.evaluationMetrics = response;
      renderEvaluationMetrics(response);

      if (metricsStatus) {
        metricsStatus.classList.remove('loading');
        metricsStatus.classList.add('success');
        metricsStatus.textContent = `✓ Evaluation complete. ${response.n_samples || 'N/A'} samples evaluated.`;
      }

      logRecommend('Evaluation metrics calculated successfully.', 'success');

    } catch (error) {
      if (metricsStatus) {
        metricsStatus.classList.remove('loading');
        metricsStatus.textContent = `Error: ${error.message}`;
      }
      logRecommend(`Error getting evaluation metrics: ${error.message}`, 'error');
    } finally {
      btnGetMetrics.classList.remove('loading');
    }
  });
}

function renderEvaluationMetrics(data) {
  const tableWrapper = document.getElementById('metrics-table-wrapper');
  const summarySection = document.getElementById('metrics-summary');

  if (!data || !data.results) return;

  const ImputePilot = data.results[PRIMARY_METHOD_KEY]?.metrics
    || data.results['ImputePilot']?.metrics
    || data.results['ImputePilot']?.metrics
    || data.results['ADART']?.metrics
    || {};
  const flaml = data.results['FLAML']?.metrics || {};

  // Define metrics to display
  const metrics = [
    { name: 'Accuracy', key: 'accuracy', higher: true },
    { name: 'Macro F1', key: 'macro_f1', higher: true },
    { name: 'MRR', key: 'mrr', higher: true },
    { name: 'Top-3 Accuracy', key: 'top_3', getter: (m) => m.top_k_accuracy?.top_3, higher: true }
  ];

  // Update table
  const tableBody = document.getElementById('metrics-table-body');
  if (tableBody) {
    tableBody.innerHTML = metrics.map(m => {
      const ImputePilotVal = m.getter ? m.getter(ImputePilot) : ImputePilot[m.key];
      const flamlVal = m.getter ? m.getter(flaml) : flaml[m.key];

      const ImputePilotNum = parseFloat(ImputePilotVal) || 0;
      const flamlNum = parseFloat(flamlVal) || 0;

      let winner = '--';
      let winnerClass = '';
      if (m.higher) {
        winner = ImputePilotNum > flamlNum ? BRAND_NAME : (flamlNum > ImputePilotNum ? 'FLAML' : 'Tie');
      } else {
        winner = ImputePilotNum < flamlNum ? BRAND_NAME : (flamlNum < ImputePilotNum ? 'FLAML' : 'Tie');
      }
      winnerClass = winner === BRAND_NAME ? 'ImputePilot' : (winner === 'FLAML' ? 'flaml' : '');

      return `
        <tr>
          <td>${m.name}</td>
          <td>${typeof ImputePilotVal === 'number' ? ImputePilotVal.toFixed(3) : (ImputePilotVal || '--')}</td>
          <td>${typeof flamlVal === 'number' ? flamlVal.toFixed(3) : (flamlVal || '--')}</td>
          <td class="winner-cell ${winnerClass}">${winner}</td>
        </tr>
      `;
    }).join('');
  }

  // Show table
  if (tableWrapper) tableWrapper.style.display = 'block';

  // Update summary
  if (summarySection) {
    summarySection.style.display = 'flex';

    // Determine overall winner
    const ImputePilotF1 = ImputePilot.macro_f1 || 0;
    const flamlF1 = flaml.macro_f1 || 0;
    const overallWinner = ImputePilotF1 > flamlF1 ? BRAND_NAME : 'FLAML';
    const improvement = ((Math.max(ImputePilotF1, flamlF1) - Math.min(ImputePilotF1, flamlF1)) / Math.min(ImputePilotF1, flamlF1) * 100).toFixed(1);

    const overallWinnerEl = document.getElementById('overall-winner');
    const improvementEl = document.getElementById('metrics-improvement');

    if (overallWinnerEl) overallWinnerEl.textContent = overallWinner;
    if (improvementEl) improvementEl.textContent = `+${improvement}%`;
  }
}

// ========== Run Downstream Button Handler ==========
const btnRunDownstream = document.getElementById('btn-run-downstream');
if (btnRunDownstream) {
  btnRunDownstream.addEventListener('click', async () => {
    const activeTask = document.querySelector('.task-card.active');
    const task = activeTask?.dataset?.task || 'forecasting';

    btnRunDownstream.classList.add('loading');
    btnRunDownstream.textContent = 'Running...';

    try {
      await loadDownstreamResults(task);
    } finally {
      btnRunDownstream.classList.remove('loading');
      btnRunDownstream.textContent = 'Run Downstream Evaluation';
    }
  });
}

// ========== Show/Hide Evaluation Metrics Panel ==========
function showEvaluationMetricsPanel() {
  const panel = document.getElementById('evaluation-metrics-panel');
  if (panel) {
    panel.style.display = 'block';
  }
}

// Show metrics panel when recommendation is complete
function onRecommendationComplete() {
  // Show FLAML card if available
  const flamlCard = document.getElementById('flaml-baseline-card');
  if (flamlCard && AppState.recommendResults?.baselines?.FLAML) {
    flamlCard.style.display = 'block';
    const flamlData = AppState.recommendResults.baselines.FLAML;

    const flamlAlgo = document.getElementById('flaml-algo');
    const flamlConfidence = document.getElementById('flaml-confidence');
    const flamlStatus = document.getElementById('flaml-status');

    if (flamlAlgo) flamlAlgo.textContent = flamlData.algo || '--';
    if (flamlConfidence) {
      flamlConfidence.textContent = flamlData.confidence
        ? `${(flamlData.confidence * 100).toFixed(1)}%` : '--';
    }
    if (flamlStatus) flamlStatus.textContent = flamlData.trained ? 'Trained' : 'Not Trained';
  }

  // Show evaluation metrics panel if ground truth is available
  if (AppState.groundTruthAvailable || AppState.evaluationMode !== 'upload') {
    showEvaluationMetricsPanel();
  }
}

// Export new functions
window.renderEvaluationMetrics = renderEvaluationMetrics;
window.showEvaluationMetricsPanel = showEvaluationMetricsPanel;
window.onRecommendationComplete = onRecommendationComplete;
window.loadDownstreamResults = loadDownstreamResults;
