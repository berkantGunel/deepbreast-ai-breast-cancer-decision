import { useState } from "react";
import {
  Upload,
  Brain,
  Loader2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Sparkles,
} from "lucide-react";
import { predictImage, type PredictionResponse } from "../services/api";

const Predict = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useTta] = useState(false); // TTA disabled by default

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
      const response = await predictImage(selectedFile, useTta);
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
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Hero Header with Gradient */}
      <div className="relative overflow-hidden bg-gradient-to-r from-red-500/20 via-pink-500/20 to-purple-500/20 border-b border-white/10">
        <div className="absolute inset-0 bg-grid-white/10"></div>
        <div className="relative max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-16 lg:py-20">
          <div className="flex items-start gap-6 animate-fade-in">
            <div className="p-4 bg-white/20 backdrop-blur-sm rounded-3xl shadow-2xl">
              <Brain className="w-12 h-12 lg:w-14 lg:h-14 text-white" />
            </div>
            <div className="flex-1">
              <h1 className="text-4xl lg:text-5xl font-bold text-white mb-3">
                AI Cancer Detection
              </h1>
              <p className="text-lg lg:text-xl text-white/90 max-w-3xl leading-relaxed">
                Advanced deep learning analysis for instant breast cancer detection.
                Upload histopathology images for accurate, AI-powered diagnosis.
              </p>
              <div className="flex items-center gap-6 mt-6">
                <div className="flex items-center gap-2 text-white/90">
                  <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
                  <span className="text-sm font-medium">92.86% Accuracy</span>
                </div>
                <div className="flex items-center gap-2 text-white/90">
                  <Sparkles className="w-4 h-4" />
                  <span className="text-sm font-medium">ResNet18 Model</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-12 lg:py-16">
        <div className="grid lg:grid-cols-5 gap-8 lg:gap-12">
          {/* Upload Section - 2 columns */}
          <div className="lg:col-span-2 space-y-8">
            <div className="animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
              {/* Section Header */}
              <div className="mb-8">
                <h2 className="text-2xl lg:text-3xl font-bold text-white mb-3">
                  Upload Medical Image
                </h2>
                <p className="text-slate-300 leading-relaxed">
                  Select a histopathology slide image for analysis
                </p>
              </div>

              {/* Upload Area */}
              <label className="block cursor-pointer group">
                <div className="relative overflow-hidden rounded-3xl border-3 border-dashed border-white/20 bg-white/5 hover:border-red-400/50 hover:bg-red-500/10 transition-all duration-300 p-10 lg:p-12">
                  <div className="flex flex-col items-center justify-center text-center">
                    <div className="relative mb-6">
                      <div className="absolute inset-0 bg-red-500/20 rounded-2xl blur-xl opacity-50 group-hover:opacity-75 transition-opacity"></div>
                      <div className="relative p-5 bg-gradient-to-br from-red-500/10 to-pink-500/10 border border-white/10 rounded-2xl">
                        <Upload className="w-12 h-12 text-red-500 group-hover:scale-110 transition-transform" />
                      </div>
                    </div>
                    <p className="text-lg font-semibold text-white mb-2">
                      Drop your image here
                    </p>
                    <p className="text-sm text-slate-300 mb-4">
                      or click to browse files
                    </p>
                    <div className="flex items-center gap-4 text-xs text-slate-400">
                      <span>PNG, JPG, TIFF</span>
                      <span>•</span>
                      <span>Max 10MB</span>
                    </div>
                  </div>
                </div>
                <input
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleFileSelect}
                />
              </label>

              {/* Image Preview */}
              {preview && (
                <div className="mt-8 animate-scale-in">
                  <p className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wide">
                    Selected Image
                  </p>
                  <div className="relative rounded-3xl overflow-hidden shadow-2xl border-4 border-white/10">
                    <img
                      src={preview}
                      alt="Preview"
                      className="w-full h-auto"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none"></div>
                  </div>
                </div>
              )}

              {/* TTA Notice - Disabled */}
              <div className="mt-8 relative rounded-3xl overflow-hidden bg-gradient-to-br from-red-500/5 via-orange-500/5 to-yellow-500/5 border border-red-400/20 p-6">
                <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/10 rounded-full blur-3xl opacity-30"></div>
                <div className="relative">
                  <div className="flex items-start gap-4">
                    <div className="p-2.5 bg-red-500/20 rounded-xl border border-red-400/30">
                      <AlertCircle className="w-5 h-5 text-red-400" />
                    </div>
                    <div className="flex-1">
                      <h4 className="text-sm font-bold text-white mb-2">
                        Test-Time Augmentation (TTA)
                      </h4>
                      <p className="text-sm text-slate-200 leading-relaxed">
                        <span className="inline-flex items-center gap-1.5 text-red-300 font-semibold mb-1">
                          <span className="text-xs">⚠️</span>
                          Currently Disabled
                        </span>
                        <br />
                        <span className="text-slate-300">
                          TTA reduces recall by 10.4% (76% vs 86%). Standard prediction provides better accuracy.
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Analyze Button */}
              <button
                onClick={handlePredict}
                disabled={!selectedFile || loading}
                className="relative w-full mt-8 group overflow-hidden rounded-2xl bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 disabled:from-gray-300 disabled:to-gray-400 transition-all duration-300 shadow-xl hover:shadow-2xl disabled:cursor-not-allowed"
              >
                <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition-opacity"></div>
                <div className="relative flex items-center justify-center gap-3 px-8 py-5">
                  {loading ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin text-white" />
                      <span className="text-lg font-semibold text-white">
                        Analyzing Image...
                      </span>
                    </>
                  ) : (
                    <>
                      <Brain className="w-6 h-6 text-white group-hover:scale-110 transition-transform" />
                      <span className="text-lg font-semibold text-white">
                        Analyze with AI
                      </span>
                    </>
                  )}
                </div>
              </button>

              {/* Error Message */}
              {error && (
                <div className="mt-6 animate-fade-in">
                  <div className="rounded-2xl bg-red-500/10 border border-red-400/30 p-5">
                    <div className="flex items-start gap-4">
                      <div className="p-2 bg-red-500/20 rounded-xl border border-red-400/30">
                        <AlertCircle className="w-5 h-5 text-red-400" />
                      </div>
                      <div>
                        <p className="font-semibold text-red-300 mb-1">Analysis Failed</p>
                        <p className="text-sm text-red-200">{error}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results Section - 3 columns */}
          <div className="lg:col-span-3 animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
            {/* Section Header */}
            <div className="mb-8">
              <h2 className="text-2xl lg:text-3xl font-bold text-white mb-3">
                Analysis Results
              </h2>
              <p className="text-slate-300 leading-relaxed">
                AI-powered diagnosis with confidence metrics
              </p>
            </div>

            {/* Empty State */}
            {!result && !loading && (
              <div className="rounded-3xl bg-white/5 border border-white/10 p-16 text-center">
                <div className="relative inline-block mb-6">
                  <div className="absolute inset-0 bg-cyan-500/20 rounded-3xl blur-2xl opacity-30"></div>
                  <div className="relative p-6 bg-gradient-to-br from-slate-900 to-slate-950 border border-white/10 rounded-3xl">
                    <Brain className="w-16 h-16 text-cyan-400" />
                  </div>
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  Waiting for Analysis
                </h3>
                <p className="text-slate-300 max-w-md mx-auto leading-relaxed">
                  Upload a histopathology image to receive instant AI-powered cancer detection results
                </p>
              </div>
            )}

            {/* Loading State */}
            {loading && (
              <div className="rounded-3xl bg-gradient-to-br from-red-500/10 to-pink-500/10 border border-red-400/30 p-16 text-center">
                <div className="relative inline-block mb-6">
                  <div className="absolute inset-0 bg-red-400/30 rounded-3xl blur-3xl opacity-40 animate-pulse"></div>
                  <div className="relative p-6 bg-slate-900 border border-white/10 rounded-3xl shadow-xl">
                    <Loader2 className="w-16 h-16 text-red-400 animate-spin" />
                  </div>
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  AI Analysis in Progress
                </h3>
                <p className="text-slate-300 max-w-md mx-auto leading-relaxed">
                  Our deep learning model is analyzing the tissue sample...
                </p>
              </div>
            )}

            {/* Results Cards */}
            {result && (
              <div className="space-y-8 animate-scale-in">
                {/* Main Diagnosis Card */}
                <div
                  className={`relative overflow-hidden rounded-[32px] border shadow-2xl ${
                    result.prediction === "Malignant"
                      ? "bg-gradient-to-br from-red-500/10 via-pink-500/10 to-rose-500/10 border-red-400/30"
                      : "bg-gradient-to-br from-green-500/10 via-emerald-500/10 to-teal-500/10 border-green-400/30"
                  }`}
                >
                  {/* Animated Background Blob */}
                  <div className="absolute top-0 right-0 w-96 h-96 rounded-full blur-3xl opacity-20 animate-pulse"
                    style={{
                      background: result.prediction === "Malignant" 
                        ? "linear-gradient(135deg, #ef4444, #ec4899)" 
                        : "linear-gradient(135deg, #10b981, #06b6d4)"
                    }}
                  ></div>
                  
                  <div className="relative p-10 lg:p-12">
                    <div className="flex flex-col space-y-6">
                      {/* Label */}
                      <div className="flex items-center gap-3">
                        <div
                          className={`p-3.5 rounded-2xl shadow-xl border ${
                            result.prediction === "Malignant"
                              ? "bg-gradient-to-br from-red-500/20 to-red-400/20 border-red-400/30"
                              : "bg-gradient-to-br from-green-500/20 to-green-400/20 border-green-400/30"
                          }`}
                        >
                          <CheckCircle
                            className={`w-9 h-9 ${
                              result.prediction === "Malignant"
                                ? "text-red-400"
                                : "text-green-400"
                            }`}
                          />
                        </div>
                        <div>
                          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5">
                            Diagnosis Result
                          </p>
                        </div>
                      </div>
                      
                      {/* Main Result */}
                      <div className="pl-2">
                        <p
                          className={`text-6xl lg:text-7xl font-black tracking-tight ${
                            result.prediction === "Malignant"
                              ? "text-red-400"
                              : "text-green-400"
                          }`}
                        >
                          {result.prediction}
                        </p>
                        <p className="text-lg text-slate-300 mt-3 font-medium">
                          {result.prediction === "Malignant" 
                            ? "Cancerous tissue detected" 
                            : "No cancer detected"}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Confidence Score Card */}
                <div className="relative overflow-hidden rounded-[32px] bg-gradient-to-br from-blue-500/10 via-indigo-500/10 to-cyan-500/10 border border-blue-400/30 shadow-2xl p-10 lg:p-12">
                  {/* Animated Background */}
                  <div className="absolute top-0 right-0 w-72 h-72 bg-blue-400/20 rounded-full blur-3xl opacity-20 animate-pulse"></div>
                  
                  <div className="relative">
                    <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 mb-8">
                      {/* Confidence Score */}
                      <div className="flex items-start gap-5">
                        <div className="p-4 bg-gradient-to-br from-blue-500/20 to-blue-400/20 border border-blue-400/30 rounded-2xl shadow-xl">
                          <TrendingUp className="w-10 h-10 text-blue-400" />
                        </div>
                        <div className="flex-1">
                          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                            Confidence Level
                          </p>
                          <p className="text-6xl lg:text-7xl font-black text-cyan-400 leading-none">
                            {result.confidence.toFixed(1)}
                            <span className="text-4xl">%</span>
                          </p>
                          <p className="text-base text-slate-300 mt-3 font-medium">
                            Model certainty score
                          </p>
                        </div>
                      </div>
                      
                      {/* TTA Badge */}
                      {result.tta_enabled && (
                        <span className="inline-flex items-center gap-2.5 bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-300 px-5 py-3 rounded-2xl font-bold text-sm shadow-lg border border-purple-400/30">
                          <Sparkles className="w-5 h-5" />
                          TTA Enhanced
                        </span>
                      )}
                    </div>
                    
                    {/* TTA Stats */}
                    {result.tta_enabled && result.prediction_std && (
                      <div className="bg-white/5 backdrop-blur-sm rounded-2xl px-6 py-4 border border-white/10">
                        <div className="flex flex-wrap items-center gap-4 text-sm">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                            <span className="font-semibold text-slate-200">
                              {result.num_augmentations} augmentations
                            </span>
                          </div>
                          <span className="text-slate-500">•</span>
                          <span className="font-semibold text-slate-300">
                            Variance: ±{Math.max(
                              result.prediction_std.benign,
                              result.prediction_std.malignant
                            ).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Probability Distribution */}
                <div className="relative overflow-hidden rounded-[32px] bg-white/5 border border-white/10 shadow-2xl p-10 lg:p-12">
                  {/* Decorative Elements */}
                  <div className="absolute top-0 left-0 w-48 h-48 bg-purple-500/10 rounded-full blur-3xl opacity-20"></div>
                  <div className="absolute bottom-0 right-0 w-48 h-48 bg-blue-500/10 rounded-full blur-3xl opacity-20"></div>
                  
                  <div className="relative">
                    {/* Header */}
                    <div className="mb-10">
                      <h3 className="text-2xl lg:text-3xl font-black text-white mb-3">
                        Probability Distribution
                      </h3>
                      <p className="text-slate-300 font-medium">
                        Class prediction probabilities
                      </p>
                    </div>

                    <div className="space-y-10">
                      {/* Benign Probability */}
                      <div className="space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-3">
                          <div className="flex items-center gap-4">
                            <div className="w-4 h-4 rounded-full bg-gradient-to-r from-green-400 to-emerald-500 shadow-lg"></div>
                            <span className="text-base lg:text-lg font-bold text-white tracking-wide">
                              BENIGN
                            </span>
                            <span className="text-sm text-slate-400 font-medium">
                              (Non-Cancerous)
                            </span>
                          </div>
                          <span className="text-3xl lg:text-4xl font-black text-green-400">
                            {result.probabilities.benign}%
                          </span>
                        </div>
                        <div className="relative h-7 bg-gradient-to-r from-white/10 to-white/5 rounded-full overflow-hidden shadow-inner border border-white/20">
                          <div
                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-green-400 via-green-500 to-emerald-500 rounded-full transition-all duration-1000 ease-out"
                            style={{ width: `${result.probabilities.benign}%` }}
                          >
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-shimmer"></div>
                          </div>
                        </div>
                      </div>

                      {/* Malignant Probability */}
                      <div className="space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-3">
                          <div className="flex items-center gap-4">
                            <div className="w-4 h-4 rounded-full bg-gradient-to-r from-red-400 to-pink-500 shadow-lg"></div>
                            <span className="text-base lg:text-lg font-bold text-white tracking-wide">
                              MALIGNANT
                            </span>
                            <span className="text-sm text-slate-400 font-medium">
                              (Cancerous)
                            </span>
                          </div>
                          <span className="text-3xl lg:text-4xl font-black text-red-400">
                            {result.probabilities.malignant}%
                          </span>
                        </div>
                        <div className="relative h-7 bg-gradient-to-r from-white/10 to-white/5 rounded-full overflow-hidden shadow-inner border border-white/20">
                          <div
                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-red-400 via-red-500 to-pink-500 rounded-full transition-all duration-1000 ease-out"
                            style={{ width: `${result.probabilities.malignant}%` }}
                          >
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-shimmer"></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Model Info Footer */}
                    <div className="mt-10 pt-8 border-t border-white/10">
                      <div className="flex items-center justify-between flex-wrap gap-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-400/30 rounded-lg">
                            <Brain className="w-5 h-5 text-purple-400" />
                          </div>
                          <div>
                            <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
                              Model
                            </p>
                            <p className="text-sm font-bold text-white">
                              ResNet18 Transfer Learning
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-400/30 rounded-lg">
                            <CheckCircle className="w-5 h-5 text-green-400" />
                          </div>
                          <div>
                            <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
                              Accuracy
                            </p>
                            <p className="text-sm font-bold text-white">
                              92.86% on Test Set
                            </p>
                          </div>
                        </div>
                      </div>
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

export default Predict;
