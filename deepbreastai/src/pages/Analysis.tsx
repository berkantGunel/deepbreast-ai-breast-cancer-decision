import { useState } from "react";
import {
  Upload,
  Eye,
  Loader2,
  AlertCircle,
  Download,
  Info,
} from "lucide-react";
import { generateGradCAM } from "../services/api";

const Analysis = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.7);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setHeatmap(null);
      setError(null);
    }
  };

  const handleGenerateHeatmap = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const blob = await generateGradCAM(selectedFile);
      const url = URL.createObjectURL(blob);
      setHeatmap(url);
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
    <div>
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-100">
        <h1 className="text-3xl font-bold text-gray-900">
          👁️ Grad-CAM Analysis
        </h1>
        <p className="text-gray-600 mt-1">
          Visualize AI attention with explainable heatmaps
        </p>
      </div>

      <div className="p-8 max-w-5xl">
        {/* Info Box */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6 flex items-start space-x-3">
          <Info className="w-5 h-5 text-blue-500 mt-0.5" />
          <div>
            <p className="text-sm text-blue-800">
              <strong>What is Grad-CAM?</strong> Gradient-weighted Class
              Activation Mapping highlights which regions of the image most
              influenced the model's prediction.
            </p>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Upload Image
            </h3>

            <label className="block cursor-pointer">
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 hover:bg-blue-50/30 transition-all">
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-sm font-medium text-gray-700">
                  Click to upload or drag and drop
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  PNG, JPG up to 10MB
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
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-700 mb-2">
                  Original Image
                </p>
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full rounded-lg border border-gray-200 shadow-sm"
                />
              </div>
            )}

            <button
              onClick={handleGenerateHeatmap}
              disabled={!selectedFile || loading}
              className="w-full mt-4 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Eye className="w-5 h-5" />
                  <span>Generate Heatmap</span>
                </>
              )}
            </button>

            {error && (
              <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-800">Error</p>
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                Heatmap Result
              </h3>
              {heatmap && (
                <button
                  onClick={handleDownload}
                  className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-700"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
              )}
            </div>

            {!heatmap && !loading && (
              <div className="bg-gray-50 rounded-xl p-8 text-center border border-gray-100">
                <Eye className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500">
                  Upload an image to generate heatmap
                </p>
              </div>
            )}

            {loading && (
              <div className="bg-gray-50 rounded-xl p-8 text-center border border-gray-100">
                <Loader2 className="w-12 h-12 text-blue-500 mx-auto mb-3 animate-spin" />
                <p className="text-gray-600 font-medium">
                  Generating heatmap...
                </p>
              </div>
            )}

            {heatmap && (
              <div className="space-y-4">
                <div className="relative">
                  <img
                    src={heatmap}
                    alt="Grad-CAM Heatmap"
                    className="w-full rounded-lg border border-gray-200 shadow-sm"
                    style={{ opacity }}
                  />
                </div>

                {/* Opacity Slider */}
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium text-gray-700">
                      Opacity
                    </label>
                    <span className="text-sm text-gray-500">
                      {Math.round(opacity * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-full appearance-none cursor-pointer accent-blue-500"
                  />
                </div>

                {/* Legend */}
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
                  <p className="text-sm font-medium text-gray-700 mb-3">
                    Color Legend
                  </p>
                  <div className="flex items-center space-x-6">
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-red-500 rounded"></div>
                      <span className="text-sm text-gray-600">
                        High attention
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-blue-500 rounded"></div>
                      <span className="text-sm text-gray-600">
                        Low attention
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;
