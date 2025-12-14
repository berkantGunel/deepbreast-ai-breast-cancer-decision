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
} from "lucide-react";
import {
  generateGradCAM,
  compareGradCAMMethods,
  type GradCAMComparisonResponse,
} from "../services/api";

const Analysis = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.7);
  const [method, setMethod] = useState<"gradcam" | "gradcam++" | "scorecam">("gradcam++");
  const [compareMode, setCompareMode] = useState(false);
  const [comparison, setComparison] = useState<GradCAMComparisonResponse | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setHeatmap(null);
      setComparison(null);
      setError(null);
    }
  };

  const handleGenerateHeatmap = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setComparison(null);
    try {
      if (compareMode) {
        const result = await compareGradCAMMethods(selectedFile);
        setComparison(result);
        setHeatmap(null);
      } else {
        const blob = await generateGradCAM(selectedFile, method);
        const url = URL.createObjectURL(blob);
        setHeatmap(url);
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
    setError(null);
  };

  return (
    <div className="page-container">
      {/* Page Header */}
      <section className="section">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 rounded-2xl">
            <Eye className="w-8 h-8 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-bold text-white">
              Grad-CAM Analysis
            </h1>
            <p className="text-slate-400 mt-1">
              Visualize AI attention with explainable heatmaps
            </p>
          </div>
        </div>
      </section>

      {/* Info Box */}
      <section className="section">
        <div className="glass-card bg-gradient-to-br from-cyan-500/5 to-blue-500/5 p-5">
          <div className="flex items-start gap-4">
            <div className="p-2 bg-cyan-500/20 border border-cyan-500/30 rounded-lg">
              <Info className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h4 className="font-semibold text-white mb-1">What is Grad-CAM?</h4>
              <p className="text-slate-400 text-sm leading-relaxed">
                Gradient-weighted Class Activation Mapping highlights which regions of the image
                most influenced the model's prediction, providing transparency and interpretability.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left Column - Upload & Settings */}
        <div className="space-y-6">
          {/* Upload Area */}
          <div className="glass-card p-6">
            <h2 className="text-xl font-semibold text-white mb-4">Upload Image</h2>

            <label className="block cursor-pointer">
              <div className="upload-zone p-10">
                <div className="flex flex-col items-center text-center">
                  <div className="p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl mb-4">
                    <Upload className="w-10 h-10 text-cyan-400" />
                  </div>
                  <p className="text-lg font-medium text-white mb-2">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-slate-400">
                    PNG, JPG or TIFF up to 10MB
                  </p>
                </div>
              </div>
              <input
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleFileSelect}
              />
            </label>
          </div>

          {/* Image Preview */}
          {preview && (
            <div className="glass-card p-6 animate-fade-in-up">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Original Image</h3>
                <button
                  onClick={clearSelection}
                  className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <XCircle className="w-5 h-5 text-slate-400 hover:text-red-400" />
                </button>
              </div>
              <div className="rounded-xl overflow-hidden border border-white/10">
                <img src={preview} alt="Preview" className="w-full h-auto" />
              </div>
            </div>
          )}

          {/* Method Selection */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              Visualization Method
            </h3>
            <div className="grid grid-cols-3 gap-2">
              {(["gradcam", "gradcam++", "scorecam"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => {
                    setMethod(m);
                    setCompareMode(false);
                  }}
                  className={`px-4 py-3 rounded-xl text-sm font-medium transition-all ${method === m && !compareMode
                      ? "bg-cyan-500 text-white"
                      : "bg-white/5 text-slate-300 hover:bg-white/10 border border-white/10"
                    }`}
                >
                  {m === "gradcam++" ? "Grad-CAM++" : m.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Compare Mode Toggle */}
          <div className="glass-card p-6">
            <label className="flex items-center justify-between cursor-pointer">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-500/20 border border-purple-500/30 rounded-lg">
                  <Grid3x3 className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <p className="font-semibold text-white">Compare All Methods</p>
                  <p className="text-sm text-slate-400">Show all three methods side-by-side</p>
                </div>
              </div>
              <input
                type="checkbox"
                checked={compareMode}
                onChange={(e) => setCompareMode(e.target.checked)}
                className="w-5 h-5"
              />
            </label>
          </div>

          {/* Generate Button */}
          <button
            onClick={handleGenerateHeatmap}
            disabled={!selectedFile || loading}
            className="w-full btn-primary py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                {compareMode ? <Grid3x3 className="w-5 h-5" /> : <Sparkles className="w-5 h-5" />}
                <span>{compareMode ? "Compare All Methods" : "Generate Heatmap"}</span>
              </>
            )}
          </button>

          {/* Error */}
          {error && (
            <div className="glass-card bg-red-500/10 border-red-500/30 p-5 animate-fade-in">
              <div className="flex items-start gap-4">
                <div className="p-2 bg-red-500/20 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                </div>
                <div>
                  <p className="font-semibold text-red-300 mb-1">Error</p>
                  <p className="text-sm text-red-200/80">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Results */}
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">
              {compareMode ? "Method Comparison" : "Heatmap Result"}
            </h2>
            {heatmap && (
              <button onClick={handleDownload} className="btn-secondary py-2 px-4">
                <Download className="w-4 h-4" />
                <span>Download</span>
              </button>
            )}
          </div>

          {/* Empty State */}
          {!heatmap && !comparison && !loading && (
            <div className="glass-card p-12 text-center">
              <div className="p-5 bg-white/5 border border-white/10 rounded-2xl w-fit mx-auto mb-5">
                <Eye className="w-12 h-12 text-slate-500" />
              </div>
              <h3 className="text-lg font-semibold text-slate-300 mb-2">No Heatmap Yet</h3>
              <p className="text-slate-500">Upload an image to generate visualization</p>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="glass-card bg-gradient-to-br from-cyan-500/5 to-blue-500/5 p-12 text-center">
              <div className="p-5 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl w-fit mx-auto mb-5">
                <Loader2 className="w-12 h-12 text-cyan-400 animate-spin" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Generating Heatmap</h3>
              <p className="text-slate-400">Computing attention regions...</p>
            </div>
          )}

          {/* Comparison View */}
          {comparison && (
            <div className="space-y-6 animate-fade-in-up">
              {Object.entries(comparison.methods).map(([methodName, data]) => (
                <div key={methodName} className="glass-card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-semibold text-white">
                      {methodName === "gradcam++" ? "Grad-CAM++" : methodName.toUpperCase()}
                    </h4>
                    <span className={`badge ${data.prediction === "Malignant" ? "badge-danger" : "badge-success"}`}>
                      {data.prediction}
                    </span>
                  </div>
                  <div className="rounded-xl overflow-hidden border border-white/10">
                    <img src={data.image} alt={`${methodName} heatmap`} className="w-full h-auto" />
                  </div>
                </div>
              ))}

              {/* Method Info */}
              <div className="glass-card p-6">
                <h4 className="font-semibold text-white mb-4">Method Comparison</h4>
                <div className="space-y-3 text-sm text-slate-300">
                  <p><strong className="text-white">Grad-CAM:</strong> Original, fast, good quality</p>
                  <p><strong className="text-white">Grad-CAM++:</strong> Improved with pixel-wise weighting, best quality</p>
                  <p><strong className="text-white">Score-CAM:</strong> Gradient-free, slowest but robust</p>
                </div>
              </div>
            </div>
          )}

          {/* Single Heatmap View */}
          {heatmap && (
            <div className="space-y-6 animate-fade-in-up">
              <div className="glass-card p-5">
                <div className="rounded-xl overflow-hidden border border-white/10">
                  <img
                    src={heatmap}
                    alt="Grad-CAM Heatmap"
                    className="w-full h-auto"
                    style={{ opacity }}
                  />
                </div>
              </div>

              {/* Opacity Slider */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold text-white">Opacity Control</h4>
                  <span className="badge badge-primary">{Math.round(opacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-2">
                  <span>Transparent</span>
                  <span>Opaque</span>
                </div>
              </div>

              {/* Legend */}
              <div className="glass-card p-6">
                <h4 className="font-semibold text-white mb-4">Heatmap Legend</h4>
                <div className="h-6 rounded-lg bg-gradient-to-r from-blue-500 via-green-500 via-yellow-500 to-red-500" />
                <div className="flex justify-between text-sm text-slate-400 mt-3">
                  <span>Low Attention</span>
                  <span>High Attention</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;
