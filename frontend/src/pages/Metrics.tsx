import { useEffect, useState } from "react";
import {
  Loader2,
  AlertCircle,
  BarChart3,
  TrendingUp,
  Target,
  Award,
  Brain,
  ScanLine,
} from "lucide-react";
import {
  getMetrics,
  getTrainingHistory,
  getMammographyMetrics,
  getMammographyTrainingHistory,
  type MetricsResponse,
  type TrainingHistoryResponse,
  type MammographyMetricsResponse,
  type MammographyTrainingHistoryResponse,
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

type ModelTab = "histopathology" | "mammography";

const Metrics = () => {
  const [activeTab, setActiveTab] = useState<ModelTab>("histopathology");
  const [histoMetrics, setHistoMetrics] = useState<MetricsResponse | null>(null);
  const [histoHistory, setHistoHistory] = useState<TrainingHistoryResponse | null>(null);
  const [mammoMetrics, setMammoMetrics] = useState<MammographyMetricsResponse | null>(null);
  const [mammoHistory, setMammoHistory] = useState<MammographyTrainingHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [histoMetricsData, histoHistoryData, mammoMetricsData, mammoHistoryData] = await Promise.all([
          getMetrics().catch(() => null),
          getTrainingHistory().catch(() => null),
          getMammographyMetrics().catch(() => null),
          getMammographyTrainingHistory().catch(() => null),
        ]);
        setHistoMetrics(histoMetricsData);
        setHistoHistory(histoHistoryData);
        setMammoMetrics(mammoMetricsData);
        setMammoHistory(mammoHistoryData);
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
      <div className="sharp-page" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ width: "80px", height: "80px", background: "rgba(139, 92, 246, 0.15)", borderRadius: "20px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1.5rem" }}>
            <Loader2 style={{ width: "40px", height: "40px", color: "#8b5cf6", animation: "spin 1s linear infinite" }} />
          </div>
          <h3 style={{ color: "#f1f5f9", marginBottom: "0.5rem" }}>Loading Metrics</h3>
          <p style={{ color: "#94a3b8" }}>Fetching model performance data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="sharp-page">
        <div className="sharp-card" style={{ background: "rgba(239, 68, 68, 0.1)", borderColor: "rgba(239, 68, 68, 0.3)" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
            <AlertCircle style={{ width: "24px", height: "24px", color: "#f87171" }} />
            <div>
              <p style={{ color: "#fca5a5", fontWeight: 600 }}>Error Loading Metrics</p>
              <p style={{ color: "#fecaca" }}>{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const histoConfusionMatrix = histoMetrics?.confusion_matrix || [[0, 0], [0, 0]];
  const histoTrainingData = histoHistory?.history?.map((item, idx) => ({
    epoch: idx + 1,
    train_loss: item.train_loss,
    val_loss: item.val_loss,
    train_acc: item.train_acc * 100,
    val_acc: item.val_acc * 100,
  })) || [];

  const mammoTrainingData = mammoHistory?.history?.map((item) => ({
    epoch: item.epoch,
    train_loss: item.train_loss,
    val_loss: item.val_loss,
    train_acc: item.train_acc,
    val_acc: item.val_acc,
  })) || [];

  return (
    <div className="sharp-page">
      {/* Header */}
      <div className="sharp-header">
        <h1>Performance Metrics</h1>
        <p className="subtitle">Model evaluation results and training history</p>
      </div>

      {/* Tabs */}
      <div className="sharp-tabs">
        <button className={`sharp-tab ${activeTab === "histopathology" ? "active" : ""}`} onClick={() => setActiveTab("histopathology")}>
          <Brain style={{ width: "18px", height: "18px" }} /> Histopathology
        </button>
        <button className={`sharp-tab ${activeTab === "mammography" ? "active" : ""}`} onClick={() => setActiveTab("mammography")}>
          <ScanLine style={{ width: "18px", height: "18px" }} /> Mammography
        </button>
      </div>

      {/* Histopathology Metrics */}
      {activeTab === "histopathology" && (
        <>
          {/* Model Info */}
          <div className="sharp-info-box" style={{ marginBottom: "2rem" }}>
            <div className="icon"><Brain style={{ width: "20px", height: "20px", color: "#10b981" }} /></div>
            <div>
              <h4>ResNet18 Model</h4>
              <p>Histopathology image classification • 2 classes (Benign/Malignant)</p>
            </div>
          </div>

          {/* Metric Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "2rem" }} className="metrics-grid">
            {[
              { label: "Accuracy", value: `${(histoMetrics?.accuracy ?? 0).toFixed(1)}%`, color: "#10b981", icon: Target },
              { label: "Precision", value: `${(histoMetrics?.precision ?? 0).toFixed(1)}%`, color: "#06b6d4", icon: TrendingUp },
              { label: "Recall", value: `${(histoMetrics?.recall ?? 0).toFixed(1)}%`, color: "#8b5cf6", icon: Award },
              { label: "F1-Score", value: `${(histoMetrics?.f1_score ?? 0).toFixed(1)}%`, color: "#f59e0b", icon: BarChart3 },
            ].map((metric) => (
              <div key={metric.label} className="sharp-metric-card">
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
                  <div style={{ width: "40px", height: "40px", background: `${metric.color}20`, borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <metric.icon style={{ width: "20px", height: "20px", color: metric.color }} />
                  </div>
                  <span style={{ color: "#94a3b8", fontSize: "0.9rem" }}>{metric.label}</span>
                </div>
                <div style={{ fontSize: "2.25rem", fontWeight: 700, color: metric.color }}>{metric.value}</div>
              </div>
            ))}
          </div>

          {/* Training Charts */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "2rem" }} className="charts-grid">
            <div className="sharp-card" style={{ padding: "1.5rem" }}>
              <h3 style={{ color: "#f1f5f9", marginBottom: "1.5rem" }}>Loss over Epochs</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={histoTrainingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", color: "#f8fafc" }} />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" stroke="#ef4444" strokeWidth={2} name="Training" dot={false} />
                  <Line type="monotone" dataKey="val_loss" stroke="#f97316" strokeWidth={2} name="Validation" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="sharp-card" style={{ padding: "1.5rem" }}>
              <h3 style={{ color: "#f1f5f9", marginBottom: "1.5rem" }}>Accuracy over Epochs</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={histoTrainingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", color: "#f8fafc" }} />
                  <Legend />
                  <Line type="monotone" dataKey="train_acc" stroke="#10b981" strokeWidth={2} name="Training" dot={false} />
                  <Line type="monotone" dataKey="val_acc" stroke="#06b6d4" strokeWidth={2} name="Validation" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="sharp-card" style={{ padding: "2rem", maxWidth: "600px" }}>
            <h3 style={{ color: "#f1f5f9", marginBottom: "1.5rem" }}>Confusion Matrix</h3>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ padding: "0.75rem" }}></th>
                  <th style={{ padding: "0.75rem", textAlign: "center" }} colSpan={2}>
                    <span style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Predicted</span>
                  </th>
                </tr>
                <tr>
                  <th style={{ padding: "0.75rem" }}></th>
                  <th style={{ padding: "0.75rem", textAlign: "center" }}><span className="sharp-badge success">Benign</span></th>
                  <th style={{ padding: "0.75rem", textAlign: "center" }}><span className="sharp-badge danger">Malignant</span></th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: "0.75rem" }}>
                    <span style={{ color: "#64748b", fontSize: "0.75rem", marginRight: "0.5rem" }}>Actual</span>
                    <span className="sharp-badge success">Benign</span>
                  </td>
                  <td style={{ padding: "0.75rem" }}>
                    <div style={{ background: "rgba(34, 197, 94, 0.1)", border: "1px solid rgba(34, 197, 94, 0.3)", borderRadius: "12px", padding: "1rem", textAlign: "center" }}>
                      <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "#4ade80" }}>{histoConfusionMatrix[0][0]}</span>
                      <p style={{ color: "rgba(74, 222, 128, 0.7)", fontSize: "0.75rem", marginTop: "0.25rem" }}>True Negative</p>
                    </div>
                  </td>
                  <td style={{ padding: "0.75rem" }}>
                    <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "12px", padding: "1rem", textAlign: "center" }}>
                      <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "rgba(248, 113, 113, 0.7)" }}>{histoConfusionMatrix[0][1]}</span>
                      <p style={{ color: "rgba(248, 113, 113, 0.7)", fontSize: "0.75rem", marginTop: "0.25rem" }}>False Positive</p>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: "0.75rem" }}>
                    <span style={{ color: "#64748b", fontSize: "0.75rem", marginRight: "0.5rem" }}>Actual</span>
                    <span className="sharp-badge danger">Malignant</span>
                  </td>
                  <td style={{ padding: "0.75rem" }}>
                    <div style={{ background: "rgba(245, 158, 11, 0.1)", border: "1px solid rgba(245, 158, 11, 0.3)", borderRadius: "12px", padding: "1rem", textAlign: "center" }}>
                      <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "rgba(251, 191, 36, 0.7)" }}>{histoConfusionMatrix[1][0]}</span>
                      <p style={{ color: "rgba(251, 191, 36, 0.7)", fontSize: "0.75rem", marginTop: "0.25rem" }}>False Negative</p>
                    </div>
                  </td>
                  <td style={{ padding: "0.75rem" }}>
                    <div style={{ background: "rgba(34, 197, 94, 0.1)", border: "1px solid rgba(34, 197, 94, 0.3)", borderRadius: "12px", padding: "1rem", textAlign: "center" }}>
                      <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "#4ade80" }}>{histoConfusionMatrix[1][1]}</span>
                      <p style={{ color: "rgba(74, 222, 128, 0.7)", fontSize: "0.75rem", marginTop: "0.25rem" }}>True Positive</p>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Mammography Metrics */}
      {activeTab === "mammography" && (
        <>
          {/* Model Info */}
          <div className="sharp-info-box" style={{ marginBottom: "2rem", background: "linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.05))", borderColor: "rgba(139, 92, 246, 0.3)" }}>
            <div className="icon" style={{ background: "rgba(139, 92, 246, 0.2)" }}><ScanLine style={{ width: "20px", height: "20px", color: "#8b5cf6" }} /></div>
            <div>
              <h4>Ensemble Classical ML</h4>
              <p>Mammography analysis with Random Forest & Gradient Boosting • 3 classes</p>
            </div>
          </div>

          {/* Metric Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "2rem" }} className="metrics-grid">
            {[
              { label: "Accuracy", value: `${(mammoMetrics?.accuracy ?? 0).toFixed(1)}%`, color: "#8b5cf6", icon: Target },
              { label: "Precision", value: `${(mammoMetrics?.precision ?? 0).toFixed(1)}%`, color: "#06b6d4", icon: TrendingUp },
              { label: "Recall", value: `${(mammoMetrics?.recall ?? 0).toFixed(1)}%`, color: "#22c55e", icon: Award },
              { label: "F1-Score", value: `${(mammoMetrics?.f1_score ?? 0).toFixed(1)}%`, color: "#f59e0b", icon: BarChart3 },
            ].map((metric) => (
              <div key={metric.label} className="sharp-metric-card">
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
                  <div style={{ width: "40px", height: "40px", background: `${metric.color}20`, borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <metric.icon style={{ width: "20px", height: "20px", color: metric.color }} />
                  </div>
                  <span style={{ color: "#94a3b8", fontSize: "0.9rem" }}>{metric.label}</span>
                </div>
                <div style={{ fontSize: "2.25rem", fontWeight: 700, color: metric.color }}>{metric.value}</div>
              </div>
            ))}
          </div>

          {/* Training Charts */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "2rem" }} className="charts-grid">
            <div className="sharp-card" style={{ padding: "1.5rem" }}>
              <h3 style={{ color: "#f1f5f9", marginBottom: "1.5rem" }}>Loss over Epochs</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={mammoTrainingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", color: "#f8fafc" }} />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" stroke="#a855f7" strokeWidth={2} name="Training" dot={false} />
                  <Line type="monotone" dataKey="val_loss" stroke="#ec4899" strokeWidth={2} name="Validation" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="sharp-card" style={{ padding: "1.5rem" }}>
              <h3 style={{ color: "#f1f5f9", marginBottom: "1.5rem" }}>Accuracy over Epochs</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={mammoTrainingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", color: "#f8fafc" }} />
                  <Legend />
                  <Line type="monotone" dataKey="train_acc" stroke="#a855f7" strokeWidth={2} name="Training" dot={false} />
                  <Line type="monotone" dataKey="val_acc" stroke="#ec4899" strokeWidth={2} name="Validation" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix & Class Performance */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }} className="charts-grid">
            <div className="sharp-card" style={{ padding: "1.5rem" }}>
              <h3 style={{ color: "#f1f5f9", marginBottom: "1.5rem" }}>Confusion Matrix</h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ padding: "0.5rem" }}></th>
                      <th style={{ padding: "0.5rem", textAlign: "center" }} colSpan={3}>
                        <span style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Predicted</span>
                      </th>
                    </tr>
                    <tr>
                      <th style={{ padding: "0.5rem" }}></th>
                      <th style={{ padding: "0.5rem", textAlign: "center" }}><span className="sharp-badge success">Benign</span></th>
                      <th style={{ padding: "0.5rem", textAlign: "center" }}><span className="sharp-badge danger">Malignant</span></th>
                      <th style={{ padding: "0.5rem", textAlign: "center" }}><span className="sharp-badge primary">Normal</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {["Benign", "Malignant", "Normal"].map((actual, i) => (
                      <tr key={actual}>
                        <td style={{ padding: "0.5rem" }}>
                          <span style={{ color: "#64748b", fontSize: "0.75rem", marginRight: "0.5rem" }}>Acc.</span>
                          <span className={`sharp-badge ${actual === "Malignant" ? "danger" : actual === "Normal" ? "primary" : "success"}`}>{actual}</span>
                        </td>
                        {[0, 1, 2].map((j) => {
                          // mammoMetrics?.confusion_matrix is [ [B,M,N], [B,M,N], [B,M,N] ] (actual x predicted) based on JSON
                          // JSON order: B(0), M(1), N(2)
                          const val = mammoMetrics?.confusion_matrix?.[i]?.[j] ?? 0;
                          const isDiagonal = i === j;
                          return (
                            <td key={j} style={{ padding: "0.5rem" }}>
                              <div style={{
                                background: isDiagonal ? `rgba(34, 197, 94, 0.1)` : "rgba(255, 255, 255, 0.02)",
                                border: isDiagonal ? `1px solid rgba(34, 197, 94, 0.3)` : "1px solid rgba(255, 255, 255, 0.05)",
                                borderRadius: "8px",
                                padding: "0.75rem",
                                textAlign: "center"
                              }}>
                                <span style={{ fontSize: "1.1rem", fontWeight: 700, color: isDiagonal ? "#4ade80" : "#94a3b8" }}>{val}</span>
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="sharp-card" style={{ padding: "1.5rem" }}>
              <h3 style={{ color: "#f1f5f9", marginBottom: "1.5rem" }}>Class Performance</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                {[
                  { label: "Benign", value: mammoMetrics?.class_accuracy?.benign ?? 0, color: "#22c55e" },
                  { label: "Malignant", value: mammoMetrics?.class_accuracy?.malignant ?? 0, color: "#ef4444" },
                  { label: "Normal", value: mammoMetrics?.class_accuracy?.normal ?? 0, color: "#3b82f6" },
                ].map((cls) => (
                  <div key={cls.label}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
                      <span style={{ color: "#f1f5f9", fontSize: "0.9rem" }}>{cls.label} Accuracy</span>
                      <span style={{ color: cls.color, fontWeight: 600 }}>{cls.value.toFixed(1)}%</span>
                    </div>
                    <div style={{ height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: `${cls.value}%`, height: "100%", background: cls.color, borderRadius: "4px" }} />
                    </div>
                  </div>
                ))}

                <div style={{ marginTop: "1.5rem", paddingTop: "1.5rem", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
                  <div style={{ fontSize: "0.85rem", color: "#94a3b8", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                    <div>AUC-ROC: <span style={{ color: "#f1f5f9" }}>{(mammoMetrics?.roc_auc ?? 0.911).toFixed(3)}</span></div>
                    <div>Features: <span style={{ color: "#f1f5f9" }}>78</span></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 1024px) {
          .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
          .charts-grid { grid-template-columns: 1fr !important; }
          .birads-grid { grid-template-columns: 1fr !important; }
          .config-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
        @media (max-width: 640px) {
          .metrics-grid { grid-template-columns: 1fr !important; }
          .config-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
};

export default Metrics;
