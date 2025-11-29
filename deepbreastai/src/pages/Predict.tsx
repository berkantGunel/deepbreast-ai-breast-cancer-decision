import { useState } from "react";
import { Upload, Brain, Loader2, AlertCircle, CheckCircle } from "lucide-react";
import { predictImage, type PredictionResponse } from "../services/api";

const Predict = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await predictImage(selectedFile);
      setResult(response);
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } };
      setError(
        error.response?.data?.detail || "Prediction failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-100">
        <h1 className="text-3xl font-bold text-gray-900">🔬 AI Prediction</h1>
        <p className="text-gray-600 mt-1">
          Upload a histopathology image for instant cancer detection
        </p>
      </div>

      <div className="p-8 max-w-5xl">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Upload Image
            </h3>

            <label className="block cursor-pointer">
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-red-400 hover:bg-red-50/30 transition-all">
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
                  Preview
                </p>
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full rounded-lg border border-gray-200 shadow-sm"
                />
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={!selectedFile || loading}
              className="w-full mt-4 bg-red-500 hover:bg-red-600 disabled:bg-gray-300 text-white py-3 px-4 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Brain className="w-5 h-5" />
                  <span>Analyze Image</span>
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
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Results
            </h3>

            {!result && !loading && (
              <div className="bg-gray-50 rounded-xl p-8 text-center border border-gray-100">
                <Brain className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500">Upload an image to see results</p>
              </div>
            )}

            {loading && (
              <div className="bg-gray-50 rounded-xl p-8 text-center border border-gray-100">
                <Loader2 className="w-12 h-12 text-red-500 mx-auto mb-3 animate-spin" />
                <p className="text-gray-600 font-medium">Analyzing image...</p>
                <p className="text-sm text-gray-500 mt-1">
                  This may take a few seconds
                </p>
              </div>
            )}

            {result && (
              <div className="space-y-4">
                {/* Diagnosis Card */}
                <div
                  className={`rounded-xl p-6 border-2 ${
                    result.prediction === "Malignant"
                      ? "bg-red-50 border-red-200"
                      : "bg-green-50 border-green-200"
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <CheckCircle
                      className={`w-6 h-6 ${
                        result.prediction === "Malignant"
                          ? "text-red-500"
                          : "text-green-500"
                      }`}
                    />
                    <span className="text-sm font-medium text-gray-600">
                      Diagnosis
                    </span>
                  </div>
                  <p
                    className={`text-3xl font-bold ${
                      result.prediction === "Malignant"
                        ? "text-red-600"
                        : "text-green-600"
                    }`}
                  >
                    {result.prediction}
                  </p>
                </div>

                {/* Confidence Card */}
                <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
                  <p className="text-sm font-medium text-gray-600 mb-2">
                    Confidence Score
                  </p>
                  <p className="text-3xl font-bold text-blue-600">
                    {result.confidence.toFixed(1)}%
                  </p>
                </div>

                {/* Probability Bars */}
                <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                  <p className="text-sm font-medium text-gray-700 mb-4">
                    Probability Distribution
                  </p>

                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">Benign</span>
                        <span className="font-semibold text-green-600">
                          {result.probabilities.benign}%
                        </span>
                      </div>
                      <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-500 rounded-full transition-all duration-500"
                          style={{ width: `${result.probabilities.benign}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">Malignant</span>
                        <span className="font-semibold text-red-600">
                          {result.probabilities.malignant}%
                        </span>
                      </div>
                      <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-red-500 rounded-full transition-all duration-500"
                          style={{
                            width: `${result.probabilities.malignant}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <p className="text-sm text-yellow-800">
                    <strong>⚠️ Disclaimer:</strong> This is an AI-assisted tool
                    for educational purposes. Always consult healthcare
                    professionals for medical decisions.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predict;
