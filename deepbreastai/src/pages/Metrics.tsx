import { useEffect, useState } from "react";
import {
  Loader2,
  AlertCircle,
  BarChart3,
  TrendingUp,
  Target,
  Award,
} from "lucide-react";
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
      <div className="w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto text-slate-50 flex items-center justify-center min-h-[60vh]">
        <div className="text-center animate-fade-in">
          <div className="p-4 bg-emerald-500/20 border border-emerald-400/30 rounded-2xl w-fit mx-auto mb-6">
            <Loader2 className="w-12 h-12 animate-spin text-emerald-400" />
          </div>
          <p className="text-title text-white">Loading metrics...</p>
          <p className="text-body text-slate-300 mt-2">
            Fetching model performance data
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto text-slate-50">
        <div className="bg-red-500/10 border border-red-400/30 rounded-2xl p-8 flex items-start space-x-4">
          <div className="p-3 bg-red-500/20 border border-red-400/30 rounded-xl">
            <AlertCircle className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <p className="text-title text-red-300">Error Loading Metrics</p>
            <p className="text-body text-red-200 mt-1">{error}</p>
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
    <div className="w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto text-slate-50">
      {/* Header */}
      <section className="section animate-fade-in">
        <div className="mb-2">
          <h1 className="text-display text-white flex items-center gap-4">
            <div className="p-3 bg-emerald-500/20 border border-emerald-400/30 rounded-2xl">
              <BarChart3 className="w-10 h-10 text-emerald-400" />
            </div>
            Performance Metrics
          </h1>
        </div>
        <p className="text-body text-slate-300 mt-4 max-w-2xl">
          Model evaluation results and training history visualization.
        </p>
      </section>

      {/* Key Metrics */}
      <section className="section">
        <h3 className="text-headline text-white mb-8">Key Metrics</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          <div
            className="metric-card animate-fade-in-up"
            style={{ animationDelay: "0.1s" }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-emerald-500/20 border border-emerald-400/30 rounded-xl">
                <Target className="w-5 h-5 text-emerald-400" />
              </div>
              <span className="text-body text-slate-300">Accuracy</span>
            </div>
            <div className="metric-value text-emerald-400">
              {(metrics?.accuracy ?? 0).toFixed(1)}%
            </div>
          </div>

          <div
            className="metric-card animate-fade-in-up"
            style={{ animationDelay: "0.15s" }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-cyan-500/20 border border-cyan-400/30 rounded-xl">
                <TrendingUp className="w-5 h-5 text-blue-600" />
              </div>
              <span className="text-body text-slate-300">Precision</span>
            </div>
            <div className="metric-value text-blue-600">
              {(metrics?.precision ?? 0).toFixed(1)}%
            </div>
          </div>

          <div
            className="metric-card animate-fade-in-up"
            style={{ animationDelay: "0.2s" }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-purple-500/20 border border-purple-400/30 rounded-xl">
                <Award className="w-5 h-5 text-purple-600" />
              </div>
              <span className="text-body text-slate-300">Recall</span>
            </div>
            <div className="metric-value text-purple-600">
              {(metrics?.recall ?? 0).toFixed(1)}%
            </div>
          </div>

          <div
            className="metric-card animate-fade-in-up"
            style={{ animationDelay: "0.25s" }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-orange-500/20 border border-orange-400/30 rounded-xl">
                <BarChart3 className="w-5 h-5 text-orange-600" />
              </div>
              <span className="text-body text-slate-300">F1-Score</span>
            </div>
            <div className="metric-value text-orange-600">
              {(metrics?.f1_score ?? 0).toFixed(1)}%
            </div>
          </div>
        </div>
      </section>

      {/* Training History Charts */}
      <section className="section">
        <h3 className="text-headline text-white mb-8">Training History</h3>
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Loss Chart */}
          <div
            className="card-elevated animate-fade-in-up"
            style={{ animationDelay: "0.3s" }}
          >
            <h4 className="text-title text-white mb-6">Loss over Epochs</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="epoch"
                  stroke="#6b7280"
                  fontSize={12}
                  tickMargin={10}
                />
                <YAxis stroke="#6b7280" fontSize={12} tickMargin={10} />
                <Tooltip
                  contentStyle={{
                    borderRadius: "16px",
                    border: "1px solid #e5e7eb",
                    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                    padding: "12px 16px",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: "20px" }} />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke="#ef4444"
                  strokeWidth={3}
                  name="Training"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#f97316"
                  strokeWidth={3}
                  name="Validation"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Accuracy Chart */}
          <div
            className="card-elevated animate-fade-in-up"
            style={{ animationDelay: "0.35s" }}
          >
            <h4 className="text-title text-white mb-6">
              Accuracy over Epochs
            </h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="epoch"
                  stroke="#6b7280"
                  fontSize={12}
                  tickMargin={10}
                />
                <YAxis
                  stroke="#6b7280"
                  fontSize={12}
                  domain={[0, 100]}
                  tickMargin={10}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: "16px",
                    border: "1px solid #e5e7eb",
                    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                    padding: "12px 16px",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: "20px" }} />
                <Line
                  type="monotone"
                  dataKey="train_acc"
                  stroke="#10b981"
                  strokeWidth={3}
                  name="Training"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_acc"
                  stroke="#059669"
                  strokeWidth={3}
                  name="Validation"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Confusion Matrix */}
      <section className="section">
        <h3 className="text-headline text-white mb-8">Confusion Matrix</h3>
        <div
          className="card-elevated max-w-2xl animate-fade-in-up"
          style={{ animationDelay: "0.4s" }}
        >
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="p-4"></th>
                  <th
                    className="p-4 text-center text-title text-white"
                    colSpan={2}
                  >
                    Predicted
                  </th>
                </tr>
                <tr>
                  <th className="p-4"></th>
                  <th className="p-4 text-center">
                    <span className="badge badge-success">Benign</span>
                  </th>
                  <th className="p-4 text-center">
                    <span className="badge badge-danger">Malignant</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="p-4 text-title text-white font-medium">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-400">Actual</span>
                      <span className="badge badge-success">Benign</span>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="bg-green-500/20 border border-green-400/30 rounded-2xl p-6 text-center">
                      <span className="text-3xl font-bold text-green-400">
                        {confusionMatrix[0][0]}
                      </span>
                      <p className="text-sm text-green-300 mt-1">
                        True Negative
                      </p>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="bg-red-500/10 border border-red-400/30 rounded-2xl p-6 text-center">
                      <span className="text-3xl font-bold text-red-300">
                        {confusionMatrix[0][1]}
                      </span>
                      <p className="text-sm text-red-200 mt-1">
                        False Positive
                      </p>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td className="p-4 text-title text-white font-medium">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-400">Actual</span>
                      <span className="badge badge-danger">Malignant</span>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="bg-orange-500/10 border border-orange-400/30 rounded-2xl p-6 text-center">
                      <span className="text-3xl font-bold text-orange-300">
                        {confusionMatrix[1][0]}
                      </span>
                      <p className="text-sm text-orange-400 mt-1">
                        False Negative
                      </p>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="bg-green-500/20 border border-green-400/30 rounded-2xl p-6 text-center">
                      <span className="text-3xl font-bold text-green-400">
                        {confusionMatrix[1][1]}
                      </span>
                      <p className="text-sm text-green-300 mt-1">
                        True Positive
                      </p>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Metrics;
