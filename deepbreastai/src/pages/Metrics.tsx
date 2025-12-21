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
      <div className="page-container flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="p-5 bg-emerald-500/10 border border-emerald-500/30 rounded-2xl w-fit mx-auto mb-5">
            <Loader2 className="w-10 h-10 text-emerald-400 animate-spin" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Loading Metrics</h3>
          <p className="text-slate-400">Fetching model performance data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="glass-card bg-red-500/10 border-red-500/30 p-6">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-red-500/20 rounded-xl">
              <AlertCircle className="w-6 h-6 text-red-400" />
            </div>
            <div>
              <p className="font-semibold text-red-300 mb-1">Error Loading Metrics</p>
              <p className="text-red-200/80">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const confusionMatrix = metrics?.confusion_matrix || [[0, 0], [0, 0]];
  const trainingData = history?.history?.map((item, idx) => ({
    epoch: idx + 1,
    train_loss: item.train_loss,
    val_loss: item.val_loss,
    train_acc: item.train_acc * 100,
    val_acc: item.val_acc * 100,
  })) || [];

  const metricCards = [
    {
      label: "Accuracy",
      value: `${(metrics?.accuracy ?? 0).toFixed(1)}%`,
      icon: Target,
      color: "text-emerald-400",
      bg: "from-emerald-500/20 to-emerald-500/10",
    },
    {
      label: "Precision",
      value: `${(metrics?.precision ?? 0).toFixed(1)}%`,
      icon: TrendingUp,
      color: "text-cyan-400",
      bg: "from-cyan-500/20 to-cyan-500/10",
    },
    {
      label: "Recall",
      value: `${(metrics?.recall ?? 0).toFixed(1)}%`,
      icon: Award,
      color: "text-purple-400",
      bg: "from-purple-500/20 to-purple-500/10",
    },
    {
      label: "F1-Score",
      value: `${(metrics?.f1_score ?? 0).toFixed(1)}%`,
      icon: BarChart3,
      color: "text-amber-400",
      bg: "from-amber-500/20 to-amber-500/10",
    },
  ];

  return (
    <div className="page-container">
      {/* Page Header */}
      <section className="section">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-gradient-to-br from-emerald-500/20 to-green-500/20 border border-emerald-500/30 rounded-2xl">
            <BarChart3 className="w-8 h-8 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-bold text-white">
              Performance Metrics
            </h1>
            <p className="text-slate-400 mt-1">
              Model evaluation results and training history
            </p>
          </div>
        </div>
      </section>

      {/* Key Metrics */}
      <section className="section">
        <h2 className="text-xl font-semibold text-white mb-6">Key Metrics</h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
          {metricCards.map((metric, index) => (
            <div
              key={metric.label}
              className="metric-card animate-fade-in-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-2 bg-gradient-to-br ${metric.bg} border border-white/10 rounded-xl`}>
                  <metric.icon className={`w-5 h-5 ${metric.color}`} />
                </div>
                <span className="text-sm text-slate-400">{metric.label}</span>
              </div>
              <div className={`text-3xl lg:text-4xl font-bold ${metric.color}`}>
                {metric.value}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Training History Charts */}
      <section className="section">
        <h2 className="text-xl font-semibold text-white mb-6">Training History</h2>
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Loss Chart */}
          <div className="glass-card p-6 animate-fade-in-up" style={{ animationDelay: "0.3s" }}>
            <h3 className="text-lg font-semibold text-white mb-6">Loss over Epochs</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis
                  dataKey="epoch"
                  stroke="#64748b"
                  fontSize={12}
                  tickMargin={10}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={12}
                  tickMargin={10}
                />
                <Tooltip
                  contentStyle={{
                    background: "#1e293b",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "12px",
                    color: "#f8fafc",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: "16px" }} />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                  name="Training"
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
          <div className="glass-card p-6 animate-fade-in-up" style={{ animationDelay: "0.4s" }}>
            <h3 className="text-lg font-semibold text-white mb-6">Accuracy over Epochs</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis
                  dataKey="epoch"
                  stroke="#64748b"
                  fontSize={12}
                  tickMargin={10}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={12}
                  domain={[0, 100]}
                  tickMargin={10}
                />
                <Tooltip
                  contentStyle={{
                    background: "#1e293b",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "12px",
                    color: "#f8fafc",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: "16px" }} />
                <Line
                  type="monotone"
                  dataKey="train_acc"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Training"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_acc"
                  stroke="#06b6d4"
                  strokeWidth={2}
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
        <h2 className="text-xl font-semibold text-white mb-6">Confusion Matrix</h2>
        <div className="glass-card p-6 lg:p-8 max-w-3xl animate-fade-in-up" style={{ animationDelay: "0.5s" }}>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="p-3"></th>
                  <th className="p-3 text-center text-sm font-semibold text-slate-300" colSpan={2}>
                    Predicted
                  </th>
                </tr>
                <tr>
                  <th className="p-3"></th>
                  <th className="p-3 text-center">
                    <span className="badge badge-success">Benign</span>
                  </th>
                  <th className="p-3 text-center">
                    <span className="badge badge-danger">Malignant</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500">Actual</span>
                      <span className="badge badge-success">Benign</span>
                    </div>
                  </td>
                  <td className="p-3">
                    <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 text-center">
                      <span className="text-2xl lg:text-3xl font-bold text-emerald-400">
                        {confusionMatrix[0][0]}
                      </span>
                      <p className="text-xs text-emerald-300/70 mt-1">True Negative</p>
                    </div>
                  </td>
                  <td className="p-3">
                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-center">
                      <span className="text-2xl lg:text-3xl font-bold text-red-400/70">
                        {confusionMatrix[0][1]}
                      </span>
                      <p className="text-xs text-red-300/70 mt-1">False Positive</p>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500">Actual</span>
                      <span className="badge badge-danger">Malignant</span>
                    </div>
                  </td>
                  <td className="p-3">
                    <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 text-center">
                      <span className="text-2xl lg:text-3xl font-bold text-amber-400/70">
                        {confusionMatrix[1][0]}
                      </span>
                      <p className="text-xs text-amber-300/70 mt-1">False Negative</p>
                    </div>
                  </td>
                  <td className="p-3">
                    <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 text-center">
                      <span className="text-2xl lg:text-3xl font-bold text-emerald-400">
                        {confusionMatrix[1][1]}
                      </span>
                      <p className="text-xs text-emerald-300/70 mt-1">True Positive</p>
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
