import { useState } from "react";
import {
  Upload,
  Brain,
  Loader2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Sparkles,
  XCircle,
  Shield,
  AlertTriangle,
  Info,
  Activity,
  FileText,
  Download,
  FileImage,
} from "lucide-react";
import {
  predictImage,
  downloadReport,
  predictDicom,
  previewDicom,
  isDicomFile,
  saveAnalysisToHistory,
  type PredictionResponse,
  type DicomPredictionResponse
} from "../services/api";

const Predict = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | DicomPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useMcDropout, setUseMcDropout] = useState(true);
  const [mcSamples, setMcSamples] = useState(30);
  const [isDicom, setIsDicom] = useState(false);
  const [dicomMetadata, setDicomMetadata] = useState<Record<string, unknown> | null>(null);
  const [savedToHistory, setSavedToHistory] = useState(false);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      setDicomMetadata(null);
      setSavedToHistory(false);

      if (isDicomFile(file)) {
        setIsDicom(true);
        try {
          const dicomPreview = await previewDicom(file);
          setPreview(dicomPreview.image);
          setDicomMetadata({
            modality: dicomPreview.modality,
            ...dicomPreview.metadata
          });
        } catch (err) {
          setPreview(null);
          console.error("DICOM preview failed:", err);
        }
      } else {
        setIsDicom(false);
        setPreview(URL.createObjectURL(file));
      }
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setSavedToHistory(false);

    try {
      let response: PredictionResponse | DicomPredictionResponse;

      if (isDicom) {
        response = await predictDicom(selectedFile, {
          useMcDropout,
          mcSamples,
          applyWindowing: true
        });
      } else {
        response = await predictImage(selectedFile, false, useMcDropout, mcSamples);
      }

      setResult(response);

      try {
        await saveAnalysisToHistory(selectedFile, {
          prediction: response.prediction,
          predicted_class: response.predicted_class,
          confidence: response.confidence,
          probabilities: response.probabilities,
          mc_dropout_enabled: response.mc_dropout_enabled,
          uncertainty: response.uncertainty,
          reliability: response.reliability,
          clinical_recommendation: response.clinical_recommendation,
        });
        setSavedToHistory(true);
      } catch (saveErr) {
        console.error("Failed to save to history:", saveErr);
      }

    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(
        error.response?.data?.detail || "Prediction failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!selectedFile) return;
    setReportLoading(true);
    try {
      await downloadReport(selectedFile, {
        includeGradcam: true,
        gradcamMethod: "gradcam++",
        mcSamples: mcSamples
      });
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(
        error.response?.data?.detail || "Report generation failed. Please try again."
      );
    } finally {
      setReportLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setIsDicom(false);
    setDicomMetadata(null);
    setSavedToHistory(false);
  };

  const getReliabilityStyle = (reliability: string | undefined) => {
    switch (reliability) {
      case "high":
        return { bg: "rgba(34, 197, 94, 0.1)", border: "rgba(34, 197, 94, 0.3)", color: "#4ade80", label: "High Reliability" };
      case "medium":
        return { bg: "rgba(245, 158, 11, 0.1)", border: "rgba(245, 158, 11, 0.3)", color: "#fbbf24", label: "Medium Reliability" };
      case "low":
        return { bg: "rgba(239, 68, 68, 0.1)", border: "rgba(239, 68, 68, 0.3)", color: "#f87171", label: "Low Reliability" };
      default:
        return { bg: "rgba(148, 163, 184, 0.1)", border: "rgba(148, 163, 184, 0.3)", color: "#94a3b8", label: "Unknown" };
    }
  };

  return (
    <div className="sharp-page">
      {/* Page Header */}
      <div className="sharp-header">
        <h1>Histopathology Analysis</h1>
        <p className="subtitle">AI-powered cancer detection with uncertainty estimation</p>
      </div>

      {/* Quick Stats */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", justifyContent: "center", marginBottom: "2rem" }}>
        {[
          { label: "95.4% Accuracy", color: "#10b981" },
          { label: "ResNet18 Model", color: "#06b6d4" },
          { label: "MC Dropout", color: "#8b5cf6" },
          { label: "PDF Reports", color: "#ec4899" },
          { label: "DICOM Support", color: "#f59e0b" },
        ].map((stat) => (
          <div key={stat.label} style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.5rem 1rem",
            background: `${stat.color}15`,
            border: `1px solid ${stat.color}40`,
            borderRadius: "20px",
            fontSize: "0.85rem",
            fontWeight: 500,
            color: stat.color
          }}>
            <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: stat.color }} />
            {stat.label}
          </div>
        ))}
      </div>

      {/* Main Content */}
      <div className="sharp-grid">
        {/* Left Column */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Upload Card */}
          <div className="sharp-card">
            <h2 style={{ color: "#f1f5f9", marginBottom: "1rem", fontSize: "1.25rem" }}>Upload Medical Image</h2>

            <label style={{ cursor: "pointer", display: "block" }}>
              <div className="sharp-drop-zone">
                <div style={{ textAlign: "center" }}>
                  <div style={{
                    width: "64px", height: "64px",
                    background: "rgba(139, 92, 246, 0.15)",
                    borderRadius: "16px",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    margin: "0 auto 1rem"
                  }}>
                    <Upload style={{ width: "28px", height: "28px", color: "#8b5cf6" }} />
                  </div>
                  <p style={{ color: "#f1f5f9", fontWeight: 500, marginBottom: "0.5rem" }}>
                    Drop your image here
                  </p>
                  <p style={{ color: "#94a3b8", fontSize: "0.9rem" }}>
                    PNG, JPG, TIFF, DICOM • Max 10MB
                  </p>
                </div>
              </div>
              <input type="file" style={{ display: "none" }} accept="image/*,.dcm,.dicom,application/dicom" onChange={handleFileSelect} />
            </label>
          </div>

          {/* Image Preview */}
          {preview && (
            <div className="sharp-card" style={{ animation: "fadeIn 0.3s ease" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                  <h3 style={{ color: "#f1f5f9", fontWeight: 600 }}>Selected Image</h3>
                  {isDicom && <span className="sharp-badge warning">DICOM</span>}
                </div>
                <button onClick={clearSelection} style={{
                  background: "none", border: "none", cursor: "pointer", padding: "0.5rem",
                  borderRadius: "8px", transition: "background 0.2s"
                }}>
                  <XCircle style={{ width: "20px", height: "20px", color: "#94a3b8" }} />
                </button>
              </div>
              <img src={preview} alt="Preview" style={{ width: "100%", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)" }} />

              {isDicom && dicomMetadata && (
                <div style={{ marginTop: "1rem", padding: "1rem", background: "rgba(245, 158, 11, 0.1)", borderRadius: "10px", border: "1px solid rgba(245, 158, 11, 0.2)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                    <FileImage style={{ width: "16px", height: "16px", color: "#f59e0b" }} />
                    <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "#fbbf24" }}>DICOM Information</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.85rem" }}>
                    <div><span style={{ color: "#94a3b8" }}>Modality:</span> <span style={{ color: "#f1f5f9" }}>{(dicomMetadata.modality as { name?: string })?.name || 'Unknown'}</span></div>
                    <div><span style={{ color: "#94a3b8" }}>Size:</span> <span style={{ color: "#f1f5f9" }}>{Number(dicomMetadata.Rows) || '?'} x {Number(dicomMetadata.Columns) || '?'}</span></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* MC Dropout Settings */}
          <div className="sharp-card">
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
              <Activity style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
              <h3 style={{ color: "#f1f5f9", fontWeight: 600 }}>Uncertainty Estimation</h3>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
              <div>
                <p style={{ color: "#f1f5f9", fontWeight: 500, marginBottom: "0.25rem" }}>MC Dropout</p>
                <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Enable Bayesian uncertainty</p>
              </div>
              <button
                onClick={() => setUseMcDropout(!useMcDropout)}
                style={{
                  width: "56px", height: "28px",
                  background: useMcDropout ? "#8b5cf6" : "#475569",
                  borderRadius: "14px", border: "none", cursor: "pointer",
                  position: "relative", transition: "background 0.3s"
                }}
              >
                <div style={{
                  width: "20px", height: "20px",
                  background: "white", borderRadius: "50%",
                  position: "absolute", top: "4px",
                  left: useMcDropout ? "32px" : "4px",
                  transition: "left 0.3s"
                }} />
              </button>
            </div>

            {useMcDropout && (
              <div style={{ animation: "fadeIn 0.3s ease" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                  <span style={{ color: "#94a3b8", fontSize: "0.9rem" }}>MC Samples</span>
                  <span className="sharp-badge primary">{mcSamples}</span>
                </div>
                <input
                  type="range" min="10" max="50" step="5"
                  value={mcSamples}
                  onChange={(e) => setMcSamples(parseInt(e.target.value))}
                  style={{ width: "100%", accentColor: "#8b5cf6" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#64748b", marginTop: "0.25rem" }}>
                  <span>Faster (10)</span>
                  <span>More Accurate (50)</span>
                </div>
              </div>
            )}
          </div>

          {/* Analyze Button */}
          <button onClick={handlePredict} disabled={!selectedFile || loading} className="sharp-btn-primary">
            {loading ? (
              <><div className="sharp-spinner" /><span>{useMcDropout ? `Running ${mcSamples} predictions...` : 'Analyzing...'}</span></>
            ) : (
              <><Brain style={{ width: "20px", height: "20px" }} /><span>Analyze with AI</span></>
            )}
          </button>

          {/* Error */}
          {error && (
            <div className="sharp-card" style={{ background: "rgba(239, 68, 68, 0.1)", borderColor: "rgba(239, 68, 68, 0.3)" }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                <AlertCircle style={{ width: "20px", height: "20px", color: "#f87171", flexShrink: 0, marginTop: "2px" }} />
                <div>
                  <p style={{ color: "#fca5a5", fontWeight: 600, marginBottom: "0.25rem" }}>Analysis Failed</p>
                  <p style={{ color: "#fecaca", fontSize: "0.9rem" }}>{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Results */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <h2 style={{ color: "#f1f5f9", fontSize: "1.25rem", fontWeight: 600 }}>Analysis Results</h2>
              {savedToHistory && (
                <span style={{ display: "flex", alignItems: "center", gap: "0.25rem", padding: "0.25rem 0.75rem", background: "rgba(34, 197, 94, 0.15)", border: "1px solid rgba(34, 197, 94, 0.3)", borderRadius: "12px", fontSize: "0.75rem", color: "#4ade80" }}>
                  <CheckCircle style={{ width: "12px", height: "12px" }} /> Saved
                </span>
              )}
            </div>
            {result && selectedFile && (
              <button onClick={handleDownloadReport} disabled={reportLoading} className="sharp-btn-secondary">
                {reportLoading ? <><div className="sharp-spinner" /><span>Generating...</span></> : <><Download style={{ width: "16px", height: "16px" }} /><span>Download PDF</span></>}
              </button>
            )}
          </div>

          {/* Empty State */}
          {!result && !loading && (
            <div className="sharp-card sharp-empty-state">
              <div className="icon-wrapper">
                <Brain style={{ width: "32px", height: "32px", color: "#8b5cf6" }} />
              </div>
              <h3>Waiting for Analysis</h3>
              <p>Upload a histopathology image to receive AI-powered cancer detection</p>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="sharp-card" style={{ textAlign: "center", padding: "4rem 2rem", background: "linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(6, 182, 212, 0.05))" }}>
              <div style={{ width: "80px", height: "80px", background: "rgba(139, 92, 246, 0.15)", borderRadius: "20px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1.5rem" }}>
                <Loader2 style={{ width: "40px", height: "40px", color: "#8b5cf6", animation: "spin 1s linear infinite" }} />
              </div>
              <h3 style={{ color: "#f1f5f9", marginBottom: "0.5rem" }}>AI Analysis in Progress</h3>
              <p style={{ color: "#94a3b8" }}>{useMcDropout ? `Running ${mcSamples} Monte Carlo samples...` : 'Analyzing tissue sample...'}</p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>
              {/* Diagnosis Card */}
              <div className="sharp-card" style={{
                padding: "2rem",
                background: result.prediction === "Malignant"
                  ? "linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05))"
                  : "linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.05))",
                borderColor: result.prediction === "Malignant" ? "rgba(239, 68, 68, 0.3)" : "rgba(34, 197, 94, 0.3)"
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
                  <div style={{
                    width: "56px", height: "56px",
                    background: result.prediction === "Malignant" ? "rgba(239, 68, 68, 0.2)" : "rgba(34, 197, 94, 0.2)",
                    borderRadius: "14px",
                    display: "flex", alignItems: "center", justifyContent: "center"
                  }}>
                    <CheckCircle style={{ width: "28px", height: "28px", color: result.prediction === "Malignant" ? "#f87171" : "#4ade80" }} />
                  </div>
                  <div>
                    <p style={{ fontSize: "0.75rem", color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "0.25rem" }}>Diagnosis Result</p>
                    <p style={{ fontSize: "2.5rem", fontWeight: 700, color: result.prediction === "Malignant" ? "#f87171" : "#4ade80" }}>
                      {result.prediction}
                    </p>
                  </div>
                </div>
                <p style={{ color: "#cbd5e1" }}>
                  {result.prediction === "Malignant" ? "Cancerous tissue detected in the sample" : "No cancer detected in the tissue sample"}
                </p>
              </div>

              {/* Confidence & Uncertainty */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                <div className="sharp-card" style={{ padding: "1.5rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
                    <div style={{ width: "40px", height: "40px", background: "rgba(6, 182, 212, 0.15)", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <TrendingUp style={{ width: "20px", height: "20px", color: "#06b6d4" }} />
                    </div>
                    <div>
                      <p style={{ fontSize: "0.75rem", color: "#94a3b8", textTransform: "uppercase" }}>Confidence</p>
                      <p style={{ fontSize: "1.75rem", fontWeight: 700, color: "#06b6d4" }}>{result.confidence}%</p>
                    </div>
                  </div>
                  <div className="sharp-progress"><div className="sharp-progress-bar cyan" style={{ width: `${result.confidence}%` }} /></div>
                </div>

                {result.uncertainty && (
                  <div className="sharp-card" style={{ padding: "1.5rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
                      <div style={{ width: "40px", height: "40px", background: "rgba(139, 92, 246, 0.15)", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <Activity style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
                      </div>
                      <div>
                        <p style={{ fontSize: "0.75rem", color: "#94a3b8", textTransform: "uppercase" }}>Uncertainty</p>
                        <p style={{ fontSize: "1.75rem", fontWeight: 700, color: "#8b5cf6" }}>{result.uncertainty.score}%</p>
                      </div>
                    </div>
                    <div className="sharp-progress"><div className="sharp-progress-bar purple" style={{ width: `${result.uncertainty.score}%` }} /></div>
                  </div>
                )}
              </div>

              {/* Reliability */}
              {result.reliability && result.clinical_recommendation && (
                <div className="sharp-card" style={{
                  padding: "1.5rem",
                  background: getReliabilityStyle(result.reliability).bg,
                  borderColor: getReliabilityStyle(result.reliability).border
                }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                    <div style={{
                      width: "48px", height: "48px",
                      background: getReliabilityStyle(result.reliability).bg,
                      borderRadius: "12px",
                      display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0
                    }}>
                      {result.reliability === "high" ? <CheckCircle style={{ width: "24px", height: "24px", color: getReliabilityStyle(result.reliability).color }} /> :
                        result.reliability === "medium" ? <AlertTriangle style={{ width: "24px", height: "24px", color: getReliabilityStyle(result.reliability).color }} /> :
                          <AlertCircle style={{ width: "24px", height: "24px", color: getReliabilityStyle(result.reliability).color }} />}
                    </div>
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
                        <h4 style={{ color: getReliabilityStyle(result.reliability).color, fontWeight: 600 }}>{getReliabilityStyle(result.reliability).label}</h4>
                        {result.n_samples && <span className="sharp-badge primary">{result.n_samples} samples</span>}
                      </div>
                      <p style={{ color: "#cbd5e1", fontSize: "0.9rem", lineHeight: 1.6 }}>{result.clinical_recommendation}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Probability Distribution */}
              <div className="sharp-card" style={{ padding: "1.5rem" }}>
                <h3 style={{ color: "#f1f5f9", fontWeight: 600, marginBottom: "1.5rem" }}>Probability Distribution</h3>

                {[
                  { label: "Benign", value: result.probabilities.benign, color: "#22c55e", std: result.std?.benign },
                  { label: "Malignant", value: result.probabilities.malignant, color: "#ef4444", std: result.std?.malignant }
                ].map((prob) => (
                  <div key={prob.label} style={{ marginBottom: "1.25rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <div style={{ width: "10px", height: "10px", borderRadius: "50%", background: prob.color }} />
                        <span style={{ color: "#f1f5f9", fontWeight: 500 }}>{prob.label}</span>
                        {prob.std && <span style={{ color: "#64748b", fontSize: "0.75rem" }}>±{prob.std}%</span>}
                      </div>
                      <span style={{ fontSize: "1.25rem", fontWeight: 700, color: prob.color }}>{prob.value}%</span>
                    </div>
                    <div className="sharp-progress">
                      <div className="sharp-progress-bar" style={{ width: `${prob.value}%`, background: `linear-gradient(90deg, ${prob.color}, ${prob.color}bb)` }} />
                    </div>
                  </div>
                ))}

                {/* Model Info */}
                <div style={{ marginTop: "1.5rem", paddingTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.1)", display: "flex", flexWrap: "wrap", gap: "1rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem", color: "#94a3b8" }}>
                    <Brain style={{ width: "16px", height: "16px", color: "#8b5cf6" }} /> ResNet18
                  </div>
                  {result.mc_dropout_enabled && (
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem", color: "#94a3b8" }}>
                      <Activity style={{ width: "16px", height: "16px", color: "#06b6d4" }} /> MC Dropout ({result.n_samples} samples)
                    </div>
                  )}
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem", color: "#94a3b8" }}>
                    <Shield style={{ width: "16px", height: "16px", color: "#10b981" }} /> 95.4% Test Accuracy
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 1024px) {
          .sharp-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
};

export default Predict;
