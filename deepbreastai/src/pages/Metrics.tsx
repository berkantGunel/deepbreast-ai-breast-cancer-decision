import { useEffect, useState } from "react";
import { Loader2, AlertCircle } from "lucide-react";
import {
  getMetrics,
  getTrainingHistory,
  type MetricsResponse,
  type TrainingHistoryResponse,
} from "../services/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const Metrics = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [history, setHistory] = useState<TrainingHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsData, historyData] = await Promise.all([
          getMetrics(),
          getTrainingHistory(),
        ]);
        setMetrics(metricsData);
        setHistory(historyData);
      } catch (err: unknown) {
        const e = err as { response?: { data?: { detail?: string } } };
        setError(e.response?.data?.detail || "Failed to load metrics");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <Loader2 className="w-10 h-10 animate-spin text-green-500 mx-auto mb-3" />
          <p className="text-gray-600">Loading metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-800">
              Error Loading Metrics
            </p>
            <p className="text-sm text-red-600">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const confusionMatrix = metrics?.confusion_matrix || [
    [0, 0],
    [0, 0],
  ];
  const trainingData =
    history?.history?.map((item, idx) => ({
      epoch: idx + 1,
      train_loss: item.train_loss,
      val_loss: item.val_loss,
      train_acc: item.train_acc * 100,
      val_acc: item.val_acc * 100,
    })) || [];

  return (
    <div>
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-100">
        <h1 className="text-3xl font-bold text-gray-900">
          📊 Performance Metrics
        </h1>
        <p className="text-gray-600 mt-1">
          Model evaluation and training history
        </p>
      </div>

      <div className="p-8 max-w-6xl">
        {/* Key Metrics */}
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Key Metrics
        </h3>
        <div className="grid grid-cols-4 gap-4 mb-8">
          <div className="bg-green-50 rounded-xl p-5 border border-green-100">
            <p className="text-sm text-gray-600 mb-1">Accuracy</p>
            <p className="text-3xl font-bold text-green-600">
              {(metrics?.accuracy ?? 0).toFixed(1)}%
            </p>
          </div>
          <div className="bg-blue-50 rounded-xl p-5 border border-blue-100">
            <p className="text-sm text-gray-600 mb-1">Precision</p>
            <p className="text-3xl font-bold text-blue-600">
              {(metrics?.precision ?? 0).toFixed(1)}%
            </p>
          </div>
          <div className="bg-purple-50 rounded-xl p-5 border border-purple-100">
            <p className="text-sm text-gray-600 mb-1">Recall</p>
            <p className="text-3xl font-bold text-purple-600">
              {(metrics?.recall ?? 0).toFixed(1)}%
            </p>
          </div>
          <div className="bg-orange-50 rounded-xl p-5 border border-orange-100">
            <p className="text-sm text-gray-600 mb-1">F1-Score</p>
            <p className="text-3xl font-bold text-orange-600">
              {(metrics?.f1_score ?? 0).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Training History Charts */}
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Training History
        </h3>
        <div className="grid lg:grid-cols-2 gap-6 mb-8">
          {/* Loss Chart */}
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-100">
            <h4 className="font-medium text-gray-700 mb-4">Loss over Epochs</h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    borderRadius: "8px",
                    border: "1px solid #e5e7eb",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                  name="Train"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#f97316"
                  strokeWidth={2}
                  name="Validation"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Accuracy Chart */}
          <div className="bg-gray-50 rounded-xl p-6 border border-gray-100">
            <h4 className="font-medium text-gray-700 mb-4">
              Accuracy over Epochs
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    borderRadius: "8px",
                    border: "1px solid #e5e7eb",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_acc"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Train"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_acc"
                  stroke="#059669"
                  strokeWidth={2}
                  name="Validation"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix */}
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Confusion Matrix
        </h3>
        <div className="bg-gray-50 rounded-xl p-6 border border-gray-100">
          <div className="max-w-md mx-auto">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div></div>
              <div className="text-sm font-medium text-gray-600 py-2">
                Pred: Benign
              </div>
              <div className="text-sm font-medium text-gray-600 py-2">
                Pred: Malignant
              </div>

              <div className="text-sm font-medium text-gray-600 py-4">
                Actual: Benign
              </div>
              <div className="bg-green-100 rounded-lg py-4 border border-green-200">
                <p className="text-2xl font-bold text-green-700">
                  {confusionMatrix[0][0]}
                </p>
                <p className="text-xs text-green-600">True Negative</p>
              </div>
              <div className="bg-red-50 rounded-lg py-4 border border-red-200">
                <p className="text-2xl font-bold text-red-600">
                  {confusionMatrix[0][1]}
                </p>
                <p className="text-xs text-red-500">False Positive</p>
              </div>

              <div className="text-sm font-medium text-gray-600 py-4">
                Actual: Malignant
              </div>
              <div className="bg-red-50 rounded-lg py-4 border border-red-200">
                <p className="text-2xl font-bold text-red-600">
                  {confusionMatrix[1][0]}
                </p>
                <p className="text-xs text-red-500">False Negative</p>
              </div>
              <div className="bg-green-100 rounded-lg py-4 border border-green-200">
                <p className="text-2xl font-bold text-green-700">
                  {confusionMatrix[1][1]}
                </p>
                <p className="text-xs text-green-600">True Positive</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Metrics;
