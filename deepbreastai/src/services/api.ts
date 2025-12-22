import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Uncertainty metrics from MC Dropout
export interface UncertaintyMetrics {
  score: number;           // 0-100%
  entropy: number;         // 0-0.693
  epistemic: number;       // 0-1
  coefficient_of_variation: number;  // %
}

// Display formatting from backend
export interface DisplayMetrics {
  uncertainty_label: string;
  reliability_color: string;
  reliability_icon: string;
  uncertainty_bar_width: string;
  confidence_bar_width: string;
}

export interface PredictionResponse {
  success: boolean;
  prediction: string;
  predicted_class: number;
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
  // MC Dropout fields
  mc_dropout_enabled?: boolean;
  n_samples?: number;
  std?: {
    benign: number;
    malignant: number;
  };
  uncertainty?: UncertaintyMetrics;
  reliability?: "high" | "medium" | "low";
  clinical_recommendation?: string;
  display?: DisplayMetrics;
  // TTA fields (legacy)
  tta_enabled?: boolean;
  prediction_std?: {
    benign: number;
    malignant: number;
  };
  num_augmentations?: number;
}

export interface MetricsResponse {
  success: boolean;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
}

export interface TrainingHistoryResponse {
  success: boolean;
  history: Array<{
    train_loss: number;
    val_loss: number;
    train_acc: number;
    val_acc: number;
  }>;
}

export interface UncertaintyInfoResponse {
  title: string;
  description: string;
  metrics: Record<string, {
    name: string;
    range: string;
    interpretation: string;
    thresholds?: Record<string, string>;
  }>;
  reliability_levels: Record<string, string>;
  clinical_use: string[];
}

export const predictImage = async (
  file: File,
  useTta: boolean = false,
  useMcDropout: boolean = true,
  mcSamples: number = 30
): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_tta", useTta.toString());
  formData.append("use_mc_dropout", useMcDropout.toString());
  formData.append("mc_samples", mcSamples.toString());

  const response = await api.post<PredictionResponse>("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};

export const getUncertaintyInfo = async (): Promise<UncertaintyInfoResponse> => {
  const response = await api.get<UncertaintyInfoResponse>("/uncertainty-info");
  return response.data;
};

export const generateGradCAM = async (
  file: File,
  method: "gradcam" | "gradcam++" | "scorecam" = "gradcam++"
): Promise<Blob> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("method", method);

  const response = await api.post("/gradcam", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    responseType: "blob",
  });

  return response.data;
};

export interface GradCAMComparisonResult {
  image: string;
  prediction: string;
}

export interface GradCAMComparisonResponse {
  success: boolean;
  methods: {
    gradcam: GradCAMComparisonResult;
    "gradcam++": GradCAMComparisonResult;
    scorecam: GradCAMComparisonResult;
  };
}

export const compareGradCAMMethods = async (
  file: File
): Promise<GradCAMComparisonResponse> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post<GradCAMComparisonResponse>(
    "/gradcam/compare",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};

export const getMetrics = async (): Promise<MetricsResponse> => {
  const response = await api.get<MetricsResponse>("/metrics");
  return response.data;
};

export const getTrainingHistory =
  async (): Promise<TrainingHistoryResponse> => {
    const response = await api.get<TrainingHistoryResponse>(
      "/training-history"
    );
    return response.data;
  };

// ========================================
// Mammography Metrics
// ========================================

export interface MammographyMetricsResponse {
  success: boolean;
  model: string;
  accuracy: number;
  test_loss: number;
  class_accuracy: {
    benign: number;
    suspicious: number;
    malignant: number;
  };
  best_val_accuracy: number;
  timestamp: string;
  classes: string[];
}

export interface MammographyTrainingHistoryResponse {
  success: boolean;
  model: string;
  config: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    image_size: number;
    classes: string[];
    mixed_precision: boolean;
  };
  history: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    train_acc: number;
    val_acc: number;
    val_class_acc?: {
      benign: number;
      suspicious: number;
      malignant: number;
    };
  }>;
}

export const getMammographyMetrics = async (): Promise<MammographyMetricsResponse> => {
  const response = await api.get<MammographyMetricsResponse>("/mammography/metrics");
  return response.data;
};

export const getMammographyTrainingHistory = async (): Promise<MammographyTrainingHistoryResponse> => {
  const response = await api.get<MammographyTrainingHistoryResponse>("/mammography/training-history");
  return response.data;
};

// PDF Report Generation
export interface ReportPreviewResponse {
  success: boolean;
  preview: {
    prediction: string;
    confidence: number;
    probabilities: {
      benign: number;
      malignant: number;
    };
    uncertainty: UncertaintyMetrics;
    reliability: string;
    clinical_recommendation: string;
    mc_samples: number;
  };
  report_options: {
    include_gradcam: boolean;
    gradcam_methods: string[];
  };
}

export const previewReport = async (
  file: File,
  mcSamples: number = 30
): Promise<ReportPreviewResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mc_samples", mcSamples.toString());

  const response = await api.post<ReportPreviewResponse>(
    "/report/preview",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};

export const generateReport = async (
  file: File,
  options: {
    includeGradcam?: boolean;
    gradcamMethod?: "gradcam" | "gradcam++" | "scorecam";
    mcSamples?: number;
    caseId?: string;
  } = {}
): Promise<Blob> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("include_gradcam", (options.includeGradcam ?? true).toString());
  formData.append("gradcam_method", options.gradcamMethod || "gradcam++");
  formData.append("mc_samples", (options.mcSamples || 30).toString());
  if (options.caseId) {
    formData.append("case_id", options.caseId);
  }

  const response = await api.post("/report/generate", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    responseType: "blob",
  });

  return response.data;
};

// Helper function to trigger PDF download
export const downloadReport = async (
  file: File,
  options: {
    includeGradcam?: boolean;
    gradcamMethod?: "gradcam" | "gradcam++" | "scorecam";
    mcSamples?: number;
    caseId?: string;
  } = {}
): Promise<void> => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("include_gradcam", (options.includeGradcam ?? true).toString());
    formData.append("gradcam_method", options.gradcamMethod || "gradcam++");
    formData.append("mc_samples", (options.mcSamples || 30).toString());
    if (options.caseId) {
      formData.append("case_id", options.caseId);
    }

    const response = await fetch(`${API_BASE_URL}/report/generate`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Report generation failed");
    }

    const blob = await response.blob();

    // Create download link
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;

    // Generate filename
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, "");
    link.download = `deepbreast_report_${options.caseId || timestamp}.pdf`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error("Download report error:", error);
    throw error;
  }
};

// ========================================
// DICOM Support
// ========================================

export interface DicomModalityInfo {
  code: string;
  name: string;
  is_mammography: boolean;
  is_supported: boolean;
  body_part: string;
}

export interface DicomMetadata {
  Modality?: string;
  Rows?: number;
  Columns?: number;
  BitsAllocated?: number;
  BitsStored?: number;
  StudyDate?: string;
  SeriesDescription?: string;
  BodyPartExamined?: string;
  IsAnonymized?: boolean;
  HasPixelData?: boolean;
  NumberOfFrames?: number;
}

export interface DicomValidateResponse {
  valid: boolean;
  message: string;
  filename: string;
  modality?: DicomModalityInfo;
  can_analyze?: boolean;
}

export interface DicomPreviewResponse {
  success: boolean;
  image: string;  // base64 data URL
  width: number;
  height: number;
  metadata: DicomMetadata;
  modality: DicomModalityInfo;
}

export interface DicomPredictionResponse extends PredictionResponse {
  source: "dicom";
  dicom_metadata: DicomMetadata;
  modality: DicomModalityInfo;
  model_note: string;
}

export interface DicomStatusResponse {
  dicom_supported: boolean;
  message: string;
  supported_modalities: string[];
  features: string[];
  note: string;
}

// Check DICOM support status
export const getDicomStatus = async (): Promise<DicomStatusResponse> => {
  const response = await api.get<DicomStatusResponse>("/dicom/status");
  return response.data;
};

// Validate DICOM file
export const validateDicom = async (file: File): Promise<DicomValidateResponse> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post<DicomValidateResponse>("/dicom/validate", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};

// Get DICOM preview
export const previewDicom = async (
  file: File,
  applyWindowing: boolean = true
): Promise<DicomPreviewResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("apply_windowing", applyWindowing.toString());

  const response = await api.post<DicomPreviewResponse>("/dicom/preview", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};

// Run prediction on DICOM file
export const predictDicom = async (
  file: File,
  options: {
    useMcDropout?: boolean;
    mcSamples?: number;
    applyWindowing?: boolean;
  } = {}
): Promise<DicomPredictionResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_mc_dropout", (options.useMcDropout ?? true).toString());
  formData.append("mc_samples", (options.mcSamples || 30).toString());
  formData.append("apply_windowing", (options.applyWindowing ?? true).toString());

  const response = await api.post<DicomPredictionResponse>("/dicom/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};

// Convert DICOM to image and download
export const convertDicom = async (
  file: File,
  outputFormat: "png" | "jpeg" = "png"
): Promise<Blob> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("output_format", outputFormat);
  formData.append("apply_windowing", "true");

  const response = await api.post("/dicom/convert", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    responseType: "blob",
  });

  return response.data;
};

// Helper to check if file is DICOM
export const isDicomFile = (file: File): boolean => {
  const name = file.name.toLowerCase();
  return name.endsWith(".dcm") || name.endsWith(".dicom") || file.type === "application/dicom";
};

// ========================================
// History & Batch Upload
// ========================================

export interface HistoryRecord {
  id: number;
  created_at: string;
  filename: string;
  file_type: string;
  prediction: string;
  predicted_class: number;
  confidence: number;
  prob_benign: number;
  prob_malignant: number;
  mc_dropout_enabled: boolean;
  uncertainty_score?: number;
  uncertainty_entropy?: number;
  uncertainty_epistemic?: number;
  reliability?: string;
  clinical_recommendation?: string;
  thumbnail?: string;
  notes?: string;
  tags?: string[];
  is_batch: boolean;
  batch_id?: string;
}

export interface HistoryResponse {
  success: boolean;
  count: number;
  offset: number;
  limit: number;
  records: HistoryRecord[];
}

export interface HistoryStats {
  total_analyses: number;
  by_prediction: Record<string, number>;
  by_reliability: Record<string, number>;
  average_confidence: number;
  daily_counts: Record<string, number>;
}

export interface BatchResult {
  id: number;
  filename: string;
  prediction: string;
  confidence: number;
  reliability?: string;
  success: boolean;
}

export interface BatchResponse {
  success: boolean;
  batch_id: string;
  summary: {
    total: number;
    success: number;
    failed: number;
    benign_count: number;
    malignant_count: number;
  };
  results: BatchResult[];
  errors: { filename: string; error: string; success: false }[];
}

// Get history records
export const getHistory = async (
  options: {
    limit?: number;
    offset?: number;
    prediction?: "Benign" | "Malignant";
    search?: string;
  } = {}
): Promise<HistoryResponse> => {
  const params = new URLSearchParams();
  if (options.limit) params.append("limit", options.limit.toString());
  if (options.offset) params.append("offset", options.offset.toString());
  if (options.prediction) params.append("prediction", options.prediction);
  if (options.search) params.append("search", options.search);

  const response = await api.get<HistoryResponse>(`/history?${params.toString()}`);
  return response.data;
};

// Get history statistics
export const getHistoryStats = async (): Promise<{ success: boolean; statistics: HistoryStats }> => {
  const response = await api.get<{ success: boolean; statistics: HistoryStats }>("/history/stats");
  return response.data;
};

// Get single record
export const getHistoryRecord = async (id: number): Promise<{ success: boolean; record: HistoryRecord }> => {
  const response = await api.get<{ success: boolean; record: HistoryRecord }>(`/history/${id}`);
  return response.data;
};

// Delete record
export const deleteHistoryRecord = async (id: number): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete<{ success: boolean; message: string }>(`/history/${id}`);
  return response.data;
};

// Clear all history
export const clearHistory = async (): Promise<{ success: boolean; message: string }> => {
  const response = await api.delete<{ success: boolean; message: string }>("/history?confirm=true");
  return response.data;
};

// Batch upload
export const batchUpload = async (
  files: File[],
  options: {
    useMcDropout?: boolean;
    mcSamples?: number;
    saveThumbnails?: boolean;
  } = {}
): Promise<BatchResponse> => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });
  formData.append("use_mc_dropout", (options.useMcDropout ?? true).toString());
  formData.append("mc_samples", (options.mcSamples || 20).toString());
  formData.append("save_thumbnails", (options.saveThumbnails ?? true).toString());

  const response = await api.post<BatchResponse>("/batch/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};

// Get batch info
export const getBatchInfo = async (batchId: string): Promise<{
  success: boolean;
  batch: {
    id: string;
    created_at: string;
    total_files: number;
    completed_files: number;
    status: string;
    summary?: {
      total: number;
      success: number;
      failed: number;
      benign_count: number;
      malignant_count: number;
    };
  };
  analyses: HistoryRecord[];
}> => {
  const response = await api.get(`/batch/${batchId}`);
  return response.data;
};

// Save single analysis to history (from Predict page)
export const saveAnalysisToHistory = async (
  file: File,
  predictionResult: {
    prediction: string;
    predicted_class: number;
    confidence: number;
    probabilities: { benign: number; malignant: number };
    mc_dropout_enabled?: boolean;
    uncertainty?: { score?: number };
    reliability?: string;
    clinical_recommendation?: string;
  },
  notes: string = ""
): Promise<{ success: boolean; record_id: number; message: string }> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("prediction", predictionResult.prediction);
  formData.append("predicted_class", predictionResult.predicted_class.toString());
  formData.append("confidence", predictionResult.confidence.toString());
  formData.append("prob_benign", predictionResult.probabilities.benign.toString());
  formData.append("prob_malignant", predictionResult.probabilities.malignant.toString());
  formData.append("mc_dropout_enabled", (predictionResult.mc_dropout_enabled || false).toString());

  if (predictionResult.uncertainty?.score !== undefined) {
    formData.append("uncertainty_score", predictionResult.uncertainty.score.toString());
  }
  if (predictionResult.reliability) {
    formData.append("reliability", predictionResult.reliability);
  }
  if (predictionResult.clinical_recommendation) {
    formData.append("clinical_recommendation", predictionResult.clinical_recommendation);
  }
  formData.append("notes", notes);
  formData.append("save_thumbnail", "true");

  const response = await api.post<{ success: boolean; record_id: number; message: string }>(
    "/history/save",
    formData,
    { headers: { "Content-Type": "multipart/form-data" } }
  );

  return response.data;
};

export default api;

// ========================================
// Mammography Analysis (BI-RADS)
// ========================================

export interface MammographyRecommendation {
  action: string;
  urgency: "low" | "medium" | "high";
  description: string;
  next_steps: string[];
}

export interface MammographyPredictionResponse {
  success: boolean;
  prediction: "Benign" | "Suspicious" | "Malignant";
  predicted_class: number;
  confidence: number;
  birads_category: string;
  probabilities: {
    benign: number;
    suspicious: number;
    malignant: number;
  };
  recommendation: MammographyRecommendation;
  model_info: {
    type: string;
    input_size: number;
    classes: string[];
  };
}

export interface MammographyInfoResponse {
  title: string;
  description: string;
  classes: Record<string, {
    birads: string;
    description: string;
    action: string;
  }>;
  dataset: {
    name: string;
    description: string;
    source: string;
  };
  model: {
    architecture: string;
    input_size: string;
    preprocessing: string;
  };
  disclaimer: string;
}

export interface MammographyHealthResponse {
  status: "healthy" | "unhealthy";
  model_loaded: boolean;
  device: string;
  model_type: string;
  error?: string;
}

// Predict mammography
export const predictMammography = async (
  file: File,
  skipValidation: boolean = false
): Promise<MammographyPredictionResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("skip_validation", skipValidation.toString());

  const response = await api.post<MammographyPredictionResponse>(
    "/mammography/predict",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
    }
  );

  return response.data;
};

// Get mammography info
export const getMammographyInfo = async (): Promise<MammographyInfoResponse> => {
  const response = await api.get<MammographyInfoResponse>("/mammography/info");
  return response.data;
};

// Check mammography health
export const getMammographyHealth = async (): Promise<MammographyHealthResponse> => {
  const response = await api.get<MammographyHealthResponse>("/mammography/health");
  return response.data;
};

// Helper to check if file is a mammogram (basic check by name)
export const isMammographyFile = (file: File): boolean => {
  const name = file.name.toLowerCase();
  // Common mammography naming patterns
  return (
    name.includes("mammo") ||
    name.includes("mlo") ||
    name.includes("cc") ||
    name.includes("breast") ||
    name.includes("birads")
  );
};

// ========================================
// Mammography Grad-CAM
// ========================================

export const generateMammographyGradCAM = async (
  file: File,
  method: "gradcam" | "gradcam++" = "gradcam++"
): Promise<Blob> => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("method", method);

  const response = await api.post("/mammography/gradcam", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    responseType: "blob",
  });

  return response.data;
};

export interface MammographyGradCAMComparisonResult {
  image: string;
  prediction: string;
  birads: string;
}

export interface MammographyGradCAMComparisonResponse {
  success: boolean;
  methods: {
    gradcam: MammographyGradCAMComparisonResult;
    "gradcam++": MammographyGradCAMComparisonResult;
  };
}

export const compareMammographyGradCAM = async (
  file: File
): Promise<MammographyGradCAMComparisonResponse> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post<MammographyGradCAMComparisonResponse>(
    "/mammography/gradcam/compare",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};
