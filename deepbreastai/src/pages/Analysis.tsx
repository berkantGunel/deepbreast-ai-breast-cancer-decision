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
  const [method, setMethod] = useState<"gradcam" | "gradcam++" | "scorecam">(
    "gradcam++"
  );
  const [compareMode, setCompareMode] = useState(false);
  const [comparison, setComparison] =
    useState<GradCAMComparisonResponse | null>(null);

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
        // Compare all methods
        const result = await compareGradCAMMethods(selectedFile);
        setComparison(result);
        setHeatmap(null);
      } else {
        // Single method
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

  return (
    <div className="w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto text-slate-50">
      {/* Header */}
      <section className="section animate-fade-in">
        <div className="mb-2">
          <h1 className="text-display text-white flex items-center gap-4">
            <div className="p-3 bg-cyan-500/20 border border-cyan-400/30 rounded-2xl">
              <Eye className="w-10 h-10 text-cyan-400" />
            </div>
            Grad-CAM Analysis
          </h1>
        </div>
        <p className="text-body text-slate-300 mt-4 max-w-2xl">
          Visualize AI attention with explainable heatmaps to understand model
          decisions.
        </p>
      </section>

      {/* Info Box */}
      <section
        className="section animate-fade-in-up"
        style={{ animationDelay: "0.1s" }}
      >
        <div className="bg-cyan-500/10 border border-cyan-400/30 rounded-2xl p-6 flex items-start space-x-4">
          <div className="p-2 bg-cyan-500/20 border border-cyan-400/30 rounded-xl">
            <Info className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <p className="text-title text-white mb-1">What is Grad-CAM?</p>
            <p className="text-body text-slate-200">
              Gradient-weighted Class Activation Mapping highlights which
              regions of the image most influenced the model's prediction,
              providing transparency and interpretability.
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="grid lg:grid-cols-2 gap-10 lg:gap-16">
          {/* Upload Section */}
          <div
            className="animate-fade-in-up"
            style={{ animationDelay: "0.15s" }}
          >
            <h3 className="text-headline text-white mb-6">Upload Image</h3>

            <label className="upload-area block cursor-pointer group">
              <div className="flex flex-col items-center justify-center py-12">
                <div className="p-4 bg-cyan-500/10 border border-cyan-400/30 rounded-2xl mb-6 group-hover:bg-cyan-500/20 transition-colors">
                  <Upload className="w-10 h-10 text-cyan-400" />
                </div>
                <p className="text-title text-white mb-2">
                  Click to upload or drag and drop
                </p>
                <p className="text-body text-slate-300">
                  PNG, JPG or TIFF up to 10MB
                </p>
              </div>
              <input
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleFileSelect}
              />
            </label>

            {preview && (
              <div className="mt-8 animate-scale-in">
                <p className="text-title text-white mb-4">Original Image</p>
                <div className="card p-4">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full rounded-xl shadow-sm"
                  />
                </div>
              </div>
            )}

            {/* Method Selection */}
            <div className="mt-8 space-y-4">
              <div className="card-lg">
                <label className="text-title text-white mb-4 block">
                  Visualization Method
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {(["gradcam", "gradcam++", "scorecam"] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => {
                        setMethod(m);
                        setCompareMode(false);
                      }}
                      className={`px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                        method === m && !compareMode
                          ? "bg-cyan-500 text-white shadow-md"
                          : "bg-white/10 text-slate-300 hover:bg-white/20"
                      }`}
                    >
                      {m === "gradcam++" ? "Grad-CAM++" : m.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Compare Mode Toggle */}
              <div className="card-lg">
                <label className="flex items-center justify-between cursor-pointer">
                  <div className="flex items-center gap-3">
                    <Grid3x3 className="w-5 h-5 text-cyan-400" />
                    <div>
                      <p className="text-title text-white">Compare All</p>
                      <p className="text-sm text-slate-300">
                        Show all three methods side-by-side
                      </p>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={compareMode}
                    onChange={(e) => setCompareMode(e.target.checked)}
                    className="w-6 h-6 text-cyan-500 rounded focus:ring-cyan-500"
                  />
                </label>
              </div>
            </div>

            <button
              onClick={handleGenerateHeatmap}
              disabled={!selectedFile || loading}
              className="btn-secondary w-full mt-8 py-4 !bg-cyan-500 hover:!bg-cyan-600"
            >
              {loading ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  {compareMode ? (
                    <Grid3x3 className="w-6 h-6" />
                  ) : (
                    <Sparkles className="w-6 h-6" />
                  )}
                  <span>
                    {compareMode ? "Compare All Methods" : "Generate Heatmap"}
                  </span>
                </>
              )}
            </button>

            {error && (
              <div className="mt-6 bg-red-500/10 border border-red-400/30 rounded-2xl p-6 flex items-start space-x-4 animate-fade-in">
                <div className="p-2 bg-red-500/20 border border-red-400/30 rounded-xl">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <p className="text-title text-red-800">Error</p>
                  <p className="text-body text-red-600 mt-1">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div
            className="animate-fade-in-up"
            style={{ animationDelay: "0.2s" }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-headline text-white">
                {compareMode ? "Method Comparison" : "Heatmap Result"}
              </h3>
              {heatmap && (
                <button
                  onClick={handleDownload}
                  className="btn-secondary !py-2 !px-4"
                >
                  <Download className="w-5 h-5" />
                  <span>Download</span>
                </button>
              )}
            </div>

            {!heatmap && !comparison && !loading && (
              <div className="card-elevated p-12 text-center">
                <div className="p-4 bg-white/5 border border-white/10 rounded-2xl w-fit mx-auto mb-6">
                  <Eye className="w-12 h-12 text-slate-400" />
                </div>
                <p className="text-title text-slate-300 mb-2">No Heatmap Yet</p>
                <p className="text-body text-slate-400">
                  Upload an image to generate visualization
                </p>
              </div>
            )}

            {loading && (
              <div className="card-elevated p-12 text-center">
                <div className="p-4 bg-cyan-500/10 border border-cyan-400/30 rounded-2xl w-fit mx-auto mb-6">
                  <Loader2 className="w-12 h-12 text-cyan-400 animate-spin" />
                </div>
                <p className="text-title text-white mb-2">
                  Generating Heatmap
                </p>
                <p className="text-body text-slate-300">
                  Computing attention regions
                </p>
              </div>
            )}

            {/* Comparison View */}
            {comparison && (
              <div className="space-y-6 animate-scale-in">
                <div className="grid grid-cols-1 gap-6">
                  {Object.entries(comparison.methods).map(
                    ([methodName, data]) => (
                      <div key={methodName} className="card p-4">
                        <div className="flex items-center justify-between mb-3">
                          <p className="text-title text-white">
                            {methodName === "gradcam++"
                              ? "Grad-CAM++"
                              : methodName.toUpperCase()}
                          </p>
                          <span
                            className={`badge ${
                              data.prediction === "Malignant"
                                ? "badge-error"
                                : "badge-success"
                            }`}
                          >
                            {data.prediction}
                          </span>
                        </div>
                        <img
                          src={data.image}
                          alt={`${methodName} heatmap`}
                          className="w-full rounded-xl shadow-sm"
                        />
                      </div>
                    )
                  )}
                </div>

                {/* Method Info */}
                <div className="bg-white/5 border border-white/10 rounded-2xl p-6 lg:p-8">
                  <p className="text-title text-white mb-3">
                    Method Comparison
                  </p>
                  <div className="space-y-2 text-sm text-slate-200">
                    <p>
                      <strong>Grad-CAM:</strong> Original, fast, good quality
                    </p>
                    <p>
                      <strong>Grad-CAM++:</strong> Improved with pixel-wise
                      weighting, best quality
                    </p>
                    <p>
                      <strong>Score-CAM:</strong> Gradient-free, slowest but
                      robust
                    </p>
                  </div>
                </div>
              </div>
            )}

            {heatmap && (
              <div className="space-y-6 animate-scale-in">
                <div className="card p-4">
                  <img
                    src={heatmap}
                    alt="Grad-CAM Heatmap"
                    className="w-full rounded-xl shadow-sm"
                    style={{ opacity }}
                  />
                </div>

                {/* Opacity Slider */}
                <div className="card-lg">
                  <div className="flex items-center justify-between mb-4">
                    <label className="text-title text-white">
                      Opacity Control
                    </label>
                    <span className="badge badge-primary">
                      {Math.round(opacity * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))}
                    className="w-full h-3 bg-white/20 rounded-full appearance-none cursor-pointer accent-cyan-500"
                  />
                  <div className="flex justify-between text-sm text-slate-400 mt-2">
                    <span>Transparent</span>
                    <span>Opaque</span>
                  </div>
                </div>

                {/* Legend */}
                <div className="card-lg">
                  <p className="text-title text-white mb-4">
                    Heatmap Legend
                  </p>
                  <div className="flex items-center gap-4">
                    <div className="flex-1 h-6 rounded-lg bg-gradient-to-r from-blue-500 via-green-500 via-yellow-500 to-red-500" />
                  </div>
                  <div className="flex justify-between text-sm text-gray-600 mt-2">
                    <span>Low Attention</span>
                    <span>High Attention</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Analysis;
