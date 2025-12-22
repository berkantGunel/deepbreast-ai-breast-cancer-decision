import { useState } from "react";
import {
  Upload,
  Eye,
  Loader2,
  AlertCircle,
  Download,
  Info,
  Sparkles,
  Grid3x3,
  XCircle,
  Brain,
  ScanLine,
} from "lucide-react";
import {
  generateGradCAM,
  compareGradCAMMethods,
  generateMammographyGradCAM,
  compareMammographyGradCAM,
  type GradCAMComparisonResponse,
  type MammographyGradCAMComparisonResponse,
} from "../services/api";

type ModelTab = "histopathology" | "mammography";

const Analysis = () => {
  const [activeTab, setActiveTab] = useState<ModelTab>("histopathology");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.7);
  const [method, setMethod] = useState<"gradcam" | "gradcam++">("gradcam++");
  const [compareMode, setCompareMode] = useState(false);
  const [comparison, setComparison] = useState<GradCAMComparisonResponse | null>(null);
  const [mammoComparison, setMammoComparison] = useState<MammographyGradCAMComparisonResponse | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setHeatmap(null);
      setComparison(null);
      setMammoComparison(null);
      setError(null);
    }
  };

  const handleGenerateHeatmap = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setComparison(null);
    setMammoComparison(null);

    try {
      if (activeTab === "histopathology") {
        // Histopathology Grad-CAM
        if (compareMode) {
          const result = await compareGradCAMMethods(selectedFile);
          setComparison(result);
          setHeatmap(null);
        } else {
          const blob = await generateGradCAM(selectedFile, method as "gradcam" | "gradcam++" | "scorecam");
          const url = URL.createObjectURL(blob);
          setHeatmap(url);
        }
      } else {
        // Mammography Grad-CAM
        if (compareMode) {
          const result = await compareMammographyGradCAM(selectedFile);
          setMammoComparison(result);
          setHeatmap(null);
        } else {
          const blob = await generateMammographyGradCAM(selectedFile, method);
          const url = URL.createObjectURL(blob);
          setHeatmap(url);
        }
      }
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(error.response?.data?.detail || "Failed to generate heatmap");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!heatmap) return;
    const a = document.createElement("a");
    a.href = heatmap;
    a.download = "gradcam_heatmap.png";
    a.click();
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    setHeatmap(null);
    setComparison(null);
    setMammoComparison(null);
    setError(null);
  };

  return (
    <div className="sharp-page">
      {/* Header */}
      <div className="sharp-header">
        <h1>Visual Analysis</h1>
        <p className="subtitle">Explainable AI visualization for model predictions</p>
      </div>

      {/* Tabs */}
      <div className="sharp-tabs">
        <button className={`sharp-tab ${activeTab === "histopathology" ? "active" : ""}`} onClick={() => { setActiveTab("histopathology"); clearSelection(); }}>
          <Brain style={{ width: "18px", height: "18px" }} /> Histopathology
        </button>
        <button className={`sharp-tab ${activeTab === "mammography" ? "active" : ""}`} onClick={() => { setActiveTab("mammography"); clearSelection(); }}>
          <ScanLine style={{ width: "18px", height: "18px" }} /> Mammography
        </button>
      </div>

      {/* Histopathology - Grad-CAM */}
      {activeTab === "histopathology" && (
        <>
          {/* Info Box */}
          <div className="sharp-info-box" style={{ marginBottom: "2rem" }}>
            <div className="icon"><Info style={{ width: "20px", height: "20px", color: "#06b6d4" }} /></div>
            <div>
              <h4>Grad-CAM for Histopathology</h4>
              <p>Gradient-weighted Class Activation Mapping highlights which regions of the histopathology image most influenced the model's prediction.</p>
            </div>
          </div>

          {/* Main Grid */}
          <div className="sharp-grid">
            {/* Left Column */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              {/* Upload */}
              <div className="sharp-card">
                <h2 style={{ color: "#f1f5f9", marginBottom: "1rem", fontSize: "1.25rem" }}>Upload Image</h2>
                <label style={{ cursor: "pointer", display: "block" }}>
                  <div className="sharp-drop-zone">
                    <div style={{ textAlign: "center" }}>
                      <div style={{ width: "64px", height: "64px", background: "rgba(6, 182, 212, 0.15)", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1rem" }}>
                        <Upload style={{ width: "28px", height: "28px", color: "#06b6d4" }} />
                      </div>
                      <p style={{ color: "#f1f5f9", fontWeight: 500, marginBottom: "0.5rem" }}>Click to upload or drag and drop</p>
                      <p style={{ color: "#94a3b8", fontSize: "0.9rem" }}>PNG, JPG or TIFF up to 10MB</p>
                    </div>
                  </div>
                  <input type="file" style={{ display: "none" }} accept="image/*" onChange={handleFileSelect} />
                </label>
              </div>

              {/* Preview */}
              {preview && (
                <div className="sharp-card" style={{ animation: "fadeIn 0.3s ease" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                    <h3 style={{ color: "#f1f5f9", fontWeight: 600 }}>Original Image</h3>
                    <button onClick={clearSelection} style={{ background: "none", border: "none", cursor: "pointer", padding: "0.5rem" }}>
                      <XCircle style={{ width: "20px", height: "20px", color: "#94a3b8" }} />
                    </button>
                  </div>
                  <img src={preview} alt="Preview" style={{ width: "100%", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)" }} />
                </div>
              )}

              {/* Method Selection */}
              <div className="sharp-card">
                <h3 style={{ color: "#f1f5f9", marginBottom: "1rem" }}>Visualization Method</h3>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem" }}>
                  {(["gradcam", "gradcam++", "scorecam"] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => { setMethod(m as "gradcam" | "gradcam++"); setCompareMode(false); }}
                      style={{
                        padding: "0.75rem",
                        background: method === m && !compareMode ? "linear-gradient(135deg, #06b6d4, #0891b2)" : "rgba(255,255,255,0.05)",
                        border: method === m && !compareMode ? "none" : "1px solid rgba(255,255,255,0.1)",
                        borderRadius: "10px",
                        color: method === m && !compareMode ? "white" : "#94a3b8",
                        fontWeight: 500,
                        cursor: "pointer",
                        transition: "all 0.3s"
                      }}
                    >
                      {m === "gradcam++" ? "Grad-CAM++" : m.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Compare Toggle */}
              <div className="sharp-card">
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                    <div style={{ width: "40px", height: "40px", background: "rgba(139, 92, 246, 0.15)", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Grid3x3 style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
                    </div>
                    <div>
                      <p style={{ color: "#f1f5f9", fontWeight: 500 }}>Compare All Methods</p>
                      <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Show all three methods side-by-side</p>
                    </div>
                  </div>
                  <input type="checkbox" checked={compareMode} onChange={(e) => setCompareMode(e.target.checked)} style={{ width: "20px", height: "20px", accentColor: "#8b5cf6" }} />
                </div>
              </div>

              {/* Generate Button */}
              <button onClick={handleGenerateHeatmap} disabled={!selectedFile || loading} className="sharp-btn-primary">
                {loading ? (
                  <><div className="sharp-spinner" /><span>Generating...</span></>
                ) : (
                  <>{compareMode ? <Grid3x3 style={{ width: "20px", height: "20px" }} /> : <Sparkles style={{ width: "20px", height: "20px" }} />}<span>{compareMode ? "Compare All Methods" : "Generate Heatmap"}</span></>
                )}
              </button>

              {/* Error */}
              {error && (
                <div className="sharp-card" style={{ background: "rgba(239, 68, 68, 0.1)", borderColor: "rgba(239, 68, 68, 0.3)" }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                    <AlertCircle style={{ width: "20px", height: "20px", color: "#f87171" }} />
                    <div>
                      <p style={{ color: "#fca5a5", fontWeight: 600, marginBottom: "0.25rem" }}>Error</p>
                      <p style={{ color: "#fecaca", fontSize: "0.9rem" }}>{error}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Results */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <h2 style={{ color: "#f1f5f9", fontSize: "1.25rem", fontWeight: 600 }}>{compareMode ? "Method Comparison" : "Heatmap Result"}</h2>
                {heatmap && (
                  <button onClick={handleDownload} className="sharp-btn-secondary">
                    <Download style={{ width: "16px", height: "16px" }} /> Download
                  </button>
                )}
              </div>

              {/* Empty State */}
              {!heatmap && !comparison && !loading && (
                <div className="sharp-card sharp-empty-state">
                  <div className="icon-wrapper"><Eye style={{ width: "32px", height: "32px", color: "#06b6d4" }} /></div>
                  <h3>No Heatmap Yet</h3>
                  <p>Upload an image to generate visualization</p>
                </div>
              )}

              {/* Loading */}
              {loading && (
                <div className="sharp-card" style={{ textAlign: "center", padding: "4rem 2rem", background: "linear-gradient(135deg, rgba(6, 182, 212, 0.05), rgba(139, 92, 246, 0.05))" }}>
                  <div style={{ width: "80px", height: "80px", background: "rgba(6, 182, 212, 0.15)", borderRadius: "20px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1.5rem" }}>
                    <Loader2 style={{ width: "40px", height: "40px", color: "#06b6d4", animation: "spin 1s linear infinite" }} />
                  </div>
                  <h3 style={{ color: "#f1f5f9", marginBottom: "0.5rem" }}>Generating Heatmap</h3>
                  <p style={{ color: "#94a3b8" }}>Computing attention regions...</p>
                </div>
              )}

              {/* Comparison View */}
              {comparison && (
                <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>
                  {Object.entries(comparison.methods).map(([methodName, data]) => (
                    <div key={methodName} className="sharp-card" style={{ padding: "1.25rem" }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                        <h4 style={{ color: "#f1f5f9", fontWeight: 600 }}>{methodName === "gradcam++" ? "Grad-CAM++" : methodName.toUpperCase()}</h4>
                        <span className={`sharp-badge ${data.prediction === "Malignant" ? "danger" : "success"}`}>{data.prediction}</span>
                      </div>
                      <img src={data.image} alt={`${methodName} heatmap`} style={{ width: "100%", borderRadius: "10px" }} />
                    </div>
                  ))}
                </div>
              )}

              {/* Single Heatmap */}
              {heatmap && (
                <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>
                  <div className="sharp-card" style={{ padding: "1.25rem" }}>
                    <img src={heatmap} alt="Grad-CAM Heatmap" style={{ width: "100%", borderRadius: "10px", opacity }} />
                  </div>

                  {/* Opacity */}
                  <div className="sharp-card" style={{ padding: "1.25rem" }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                      <h4 style={{ color: "#f1f5f9", fontWeight: 600 }}>Opacity Control</h4>
                      <span className="sharp-badge primary">{Math.round(opacity * 100)}%</span>
                    </div>
                    <input type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#8b5cf6" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#64748b", marginTop: "0.25rem" }}>
                      <span>Transparent</span><span>Opaque</span>
                    </div>
                  </div>

                  {/* Legend */}
                  <div className="sharp-card" style={{ padding: "1.25rem" }}>
                    <h4 style={{ color: "#f1f5f9", fontWeight: 600, marginBottom: "1rem" }}>Heatmap Legend</h4>
                    <div style={{ height: "20px", borderRadius: "10px", background: "linear-gradient(90deg, #3b82f6, #22c55e, #eab308, #ef4444)" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "#94a3b8", marginTop: "0.75rem" }}>
                      <span>Low Attention</span><span>High Attention</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Mammography - Grad-CAM */}
      {activeTab === "mammography" && (
        <>
          {/* Info Box */}
          <div className="sharp-info-box" style={{ marginBottom: "2rem", background: "linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.05))", borderColor: "rgba(139, 92, 246, 0.3)" }}>
            <div className="icon" style={{ background: "rgba(139, 92, 246, 0.2)" }}><Info style={{ width: "20px", height: "20px", color: "#8b5cf6" }} /></div>
            <div>
              <h4>Grad-CAM for Mammography</h4>
              <p>Gradient-weighted Class Activation Mapping highlights which regions of the mammogram most influenced the BI-RADS classification.</p>
            </div>
          </div>

          {/* Main Grid */}
          <div className="sharp-grid">
            {/* Left Column */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              {/* Upload */}
              <div className="sharp-card">
                <h2 style={{ color: "#f1f5f9", marginBottom: "1rem", fontSize: "1.25rem" }}>Upload Mammogram</h2>
                <label style={{ cursor: "pointer", display: "block" }}>
                  <div className="sharp-drop-zone" style={{ borderColor: "rgba(139, 92, 246, 0.3)" }}>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ width: "64px", height: "64px", background: "rgba(139, 92, 246, 0.15)", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1rem" }}>
                        <ScanLine style={{ width: "28px", height: "28px", color: "#8b5cf6" }} />
                      </div>
                      <p style={{ color: "#f1f5f9", fontWeight: 500, marginBottom: "0.5rem" }}>Click to upload mammography image</p>
                      <p style={{ color: "#94a3b8", fontSize: "0.9rem" }}>PNG, JPG or DICOM up to 10MB</p>
                    </div>
                  </div>
                  <input type="file" style={{ display: "none" }} accept="image/*" onChange={handleFileSelect} />
                </label>
              </div>

              {/* Preview */}
              {preview && (
                <div className="sharp-card" style={{ animation: "fadeIn 0.3s ease" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                    <h3 style={{ color: "#f1f5f9", fontWeight: 600 }}>Original Mammogram</h3>
                    <button onClick={clearSelection} style={{ background: "none", border: "none", cursor: "pointer", padding: "0.5rem" }}>
                      <XCircle style={{ width: "20px", height: "20px", color: "#94a3b8" }} />
                    </button>
                  </div>
                  <img src={preview} alt="Preview" style={{ width: "100%", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)" }} />
                </div>
              )}

              {/* Method Selection */}
              <div className="sharp-card">
                <h3 style={{ color: "#f1f5f9", marginBottom: "1rem" }}>Visualization Method</h3>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "0.5rem" }}>
                  {(["gradcam", "gradcam++"] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => { setMethod(m); setCompareMode(false); }}
                      style={{
                        padding: "0.75rem",
                        background: method === m && !compareMode ? "linear-gradient(135deg, #8b5cf6, #a855f7)" : "rgba(255,255,255,0.05)",
                        border: method === m && !compareMode ? "none" : "1px solid rgba(255,255,255,0.1)",
                        borderRadius: "10px",
                        color: method === m && !compareMode ? "white" : "#94a3b8",
                        fontWeight: 500,
                        cursor: "pointer",
                        transition: "all 0.3s"
                      }}
                    >
                      {m === "gradcam++" ? "Grad-CAM++" : "Grad-CAM"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Compare Toggle */}
              <div className="sharp-card">
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                    <div style={{ width: "40px", height: "40px", background: "rgba(236, 72, 153, 0.15)", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Grid3x3 style={{ width: "20px", height: "20px", color: "#ec4899" }} />
                    </div>
                    <div>
                      <p style={{ color: "#f1f5f9", fontWeight: 500 }}>Compare Methods</p>
                      <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Show both methods side-by-side</p>
                    </div>
                  </div>
                  <input type="checkbox" checked={compareMode} onChange={(e) => setCompareMode(e.target.checked)} style={{ width: "20px", height: "20px", accentColor: "#ec4899" }} />
                </div>
              </div>

              {/* Generate Button */}
              <button onClick={handleGenerateHeatmap} disabled={!selectedFile || loading} className="sharp-btn-primary" style={{ background: "linear-gradient(135deg, #8b5cf6, #ec4899)" }}>
                {loading ? (
                  <><div className="sharp-spinner" /><span>Generating...</span></>
                ) : (
                  <>{compareMode ? <Grid3x3 style={{ width: "20px", height: "20px" }} /> : <Sparkles style={{ width: "20px", height: "20px" }} />}<span>{compareMode ? "Compare Methods" : "Generate Heatmap"}</span></>
                )}
              </button>

              {/* Error */}
              {error && (
                <div className="sharp-card" style={{ background: "rgba(239, 68, 68, 0.1)", borderColor: "rgba(239, 68, 68, 0.3)" }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                    <AlertCircle style={{ width: "20px", height: "20px", color: "#f87171" }} />
                    <div>
                      <p style={{ color: "#fca5a5", fontWeight: 600, marginBottom: "0.25rem" }}>Error</p>
                      <p style={{ color: "#fecaca", fontSize: "0.9rem" }}>{error}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Results */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <h2 style={{ color: "#f1f5f9", fontSize: "1.25rem", fontWeight: 600 }}>{compareMode ? "Method Comparison" : "Heatmap Result"}</h2>
                {heatmap && (
                  <button onClick={handleDownload} className="sharp-btn-secondary">
                    <Download style={{ width: "16px", height: "16px" }} /> Download
                  </button>
                )}
              </div>

              {/* Empty State */}
              {!heatmap && !mammoComparison && !loading && (
                <div className="sharp-card sharp-empty-state">
                  <div className="icon-wrapper" style={{ background: "rgba(139, 92, 246, 0.15)" }}><Eye style={{ width: "32px", height: "32px", color: "#8b5cf6" }} /></div>
                  <h3>No Heatmap Yet</h3>
                  <p>Upload a mammogram to generate visualization</p>
                </div>
              )}

              {/* Loading */}
              {loading && (
                <div className="sharp-card" style={{ textAlign: "center", padding: "4rem 2rem", background: "linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(236, 72, 153, 0.05))" }}>
                  <div style={{ width: "80px", height: "80px", background: "rgba(139, 92, 246, 0.15)", borderRadius: "20px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 1.5rem" }}>
                    <Loader2 style={{ width: "40px", height: "40px", color: "#8b5cf6", animation: "spin 1s linear infinite" }} />
                  </div>
                  <h3 style={{ color: "#f1f5f9", marginBottom: "0.5rem" }}>Generating Heatmap</h3>
                  <p style={{ color: "#94a3b8" }}>Analyzing mammography regions...</p>
                </div>
              )}

              {/* Mammography Comparison View */}
              {mammoComparison && (
                <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>
                  {Object.entries(mammoComparison.methods).map(([methodName, data]) => (
                    <div key={methodName} className="sharp-card" style={{ padding: "1.25rem" }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                        <h4 style={{ color: "#f1f5f9", fontWeight: 600 }}>{methodName === "gradcam++" ? "Grad-CAM++" : "Grad-CAM"}</h4>
                        <div style={{ display: "flex", gap: "0.5rem" }}>
                          <span className={`sharp-badge ${data.prediction === "Malignant" ? "danger" : data.prediction === "Suspicious" ? "warning" : "success"}`}>{data.prediction}</span>
                          <span className="sharp-badge primary">{data.birads}</span>
                        </div>
                      </div>
                      <img src={data.image} alt={`${methodName} heatmap`} style={{ width: "100%", borderRadius: "10px" }} />
                    </div>
                  ))}
                </div>
              )}

              {/* Single Heatmap */}
              {heatmap && !mammoComparison && (
                <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>
                  <div className="sharp-card" style={{ padding: "1.25rem" }}>
                    <img src={heatmap} alt="Mammography Grad-CAM Heatmap" style={{ width: "100%", borderRadius: "10px", opacity }} />
                  </div>

                  {/* Opacity */}
                  <div className="sharp-card" style={{ padding: "1.25rem" }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                      <h4 style={{ color: "#f1f5f9", fontWeight: 600 }}>Opacity Control</h4>
                      <span className="sharp-badge primary">{Math.round(opacity * 100)}%</span>
                    </div>
                    <input type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#8b5cf6" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#64748b", marginTop: "0.25rem" }}>
                      <span>Transparent</span><span>Opaque</span>
                    </div>
                  </div>

                  {/* Legend */}
                  <div className="sharp-card" style={{ padding: "1.25rem" }}>
                    <h4 style={{ color: "#f1f5f9", fontWeight: 600, marginBottom: "1rem" }}>Heatmap Legend</h4>
                    <div style={{ height: "20px", borderRadius: "10px", background: "linear-gradient(90deg, #3b82f6, #22c55e, #eab308, #ef4444)" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "#94a3b8", marginTop: "0.75rem" }}>
                      <span>Low Attention</span><span>High Attention</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}

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

export default Analysis;
