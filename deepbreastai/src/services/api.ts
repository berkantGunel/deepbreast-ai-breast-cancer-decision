import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface PredictionResponse {
  success: boolean;
  prediction: string;
  predicted_class: number;
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
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

export const predictImage = async (file: File): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post<PredictionResponse>("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};

export const generateGradCAM = async (file: File): Promise<Blob> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/gradcam", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    responseType: "blob",
  });

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

export default api;
