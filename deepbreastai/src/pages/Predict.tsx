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

      // Check if DICOM file
      if (isDicomFile(file)) {
        setIsDicom(true);
        // Get DICOM preview
        try {
          const dicomPreview = await previewDicom(file);
          setPreview(dicomPreview.image);
          setDicomMetadata({
            modality: dicomPreview.modality,
            ...dicomPreview.metadata
          });
        } catch (err) {
          // If preview fails, no image preview
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
        // Use DICOM-specific prediction
        response = await predictDicom(selectedFile, {
          useMcDropout,
          mcSamples,
          applyWindowing: true
        });
      } else {
        // Standard image prediction
        response = await predictImage(selectedFile, false, useMcDropout, mcSamples);
      }

      setResult(response);

      // Auto-save to history
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
        // Don't show error - analysis still succeeded
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

  // Helper to get reliability styling
  const getReliabilityStyle = (reliability: string | undefined) => {
    switch (reliability) {
      case "high":
        return {
          bg: "bg-emerald-500/10",
          border: "border-emerald-500/30",
          text: "text-emerald-400",
          icon: CheckCircle,
          label: "High Reliability"
        };
      case "medium":
        return {
          bg: "bg-amber-500/10",
          border: "border-amber-500/30",
          text: "text-amber-400",
          icon: AlertTriangle,
          label: "Medium Reliability"
        };
      case "low":
        return {
          bg: "bg-red-500/10",
          border: "border-red-500/30",
          text: "text-red-400",
          icon: AlertCircle,
          label: "Low Reliability"
        };
      default:
        return {
          bg: "bg-slate-500/10",
          border: "border-slate-500/30",
          text: "text-slate-400",
          icon: Info,
          label: "Unknown"
        };
    }
  };

  return (
    <div className="page-container">
      {/* Page Header */}
      <section className="section">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30 rounded-2xl">
            <Brain className="w-8 h-8 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-bold text-white">
              AI Cancer Detection
            </h1>
            <p className="text-slate-400 mt-1">
              Upload histopathology images for AI-powered analysis with uncertainty estimation
            </p>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="flex flex-wrap gap-4 mt-6">
          <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-sm font-medium text-emerald-400">92.86% Accuracy</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-xl">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-cyan-400">ResNet18 Model</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-purple-500/10 border border-purple-500/30 rounded-xl">
            <Activity className="w-4 h-4 text-purple-400" />
            <span className="text-sm font-medium text-purple-400">MC Dropout Uncertainty</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-pink-500/10 border border-pink-500/30 rounded-xl">
            <FileText className="w-4 h-4 text-pink-400" />
            <span className="text-sm font-medium text-pink-400">PDF Reports</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-orange-500/10 border border-orange-500/30 rounded-xl">
            <FileImage className="w-4 h-4 text-orange-400" />
            <span className="text-sm font-medium text-orange-400">DICOM Support</span>
          </div>
        </div>
      </section>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          {/* Upload Area */}
          <div className="glass-card p-6">
            <h2 className="text-xl font-semibold text-white mb-4">
              Upload Medical Image
            </h2>

            <label className="block cursor-pointer">
              <div className="upload-zone p-10">
                <div className="flex flex-col items-center text-center">
                  <div className="p-4 bg-white/5 border border-white/10 rounded-2xl mb-4">
                    <Upload className="w-10 h-10 text-slate-400" />
                  </div>
                  <p className="text-lg font-medium text-white mb-2">
                    Drop your image here
                  </p>
                  <p className="text-sm text-slate-400 mb-3">
                    or click to browse files
                  </p>
                  <div className="flex items-center gap-3 text-xs text-slate-500">
                    <span>PNG, JPG, TIFF, DICOM</span>
                    <span>•</span>
                    <span>Max 10MB</span>
                  </div>
                </div>
              </div>
              <input
                type="file"
                className="hidden"
                accept="image/*,.dcm,.dicom,application/dicom"
                onChange={handleFileSelect}
              />
            </label>
          </div>

          {/* Image Preview */}
          {preview && (
            <div className="glass-card p-6 animate-fade-in-up">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold text-white">Selected Image</h3>
                  {isDicom && (
                    <span className="badge badge-warning text-xs">DICOM</span>
                  )}
                </div>
                <button
                  onClick={clearSelection}
                  className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <XCircle className="w-5 h-5 text-slate-400 hover:text-red-400" />
                </button>
              </div>
              <div className="relative rounded-2xl overflow-hidden border border-white/10">
                <img src={preview} alt="Preview" className="w-full h-auto" />
              </div>

              {/* DICOM Metadata */}
              {isDicom && dicomMetadata && (
                <div className="mt-4 p-4 bg-orange-500/5 border border-orange-500/20 rounded-xl">
                  <div className="flex items-center gap-2 mb-3">
                    <FileImage className="w-4 h-4 text-orange-400" />
                    <span className="text-sm font-medium text-orange-400">DICOM Information</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-slate-500">Modality:</span>
                      <span className="text-white ml-2">
                        {(dicomMetadata.modality as { name?: string })?.name || 'Unknown'}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500">Size:</span>
                      <span className="text-white ml-2">
                        {Number(dicomMetadata.Rows) || '?'} x {Number(dicomMetadata.Columns) || '?'}
                      </span>
                    </div>
                    {dicomMetadata.StudyDate && (
                      <div>
                        <span className="text-slate-500">Study Date:</span>
                        <span className="text-white ml-2">{String(dicomMetadata.StudyDate)}</span>
                      </div>
                    )}
                    {dicomMetadata.BodyPartExamined && (
                      <div>
                        <span className="text-slate-500">Body Part:</span>
                        <span className="text-white ml-2">{String(dicomMetadata.BodyPartExamined)}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* MC Dropout Settings */}
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-4">
              <Activity className="w-5 h-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-white">Uncertainty Estimation</h3>
            </div>

            {/* MC Dropout Toggle */}
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="font-medium text-white">MC Dropout</p>
                <p className="text-sm text-slate-400">Enable Bayesian uncertainty estimation</p>
              </div>
              <button
                onClick={() => setUseMcDropout(!useMcDropout)}
                className={`relative w-14 h-7 rounded-full transition-colors ${useMcDropout ? 'bg-purple-500' : 'bg-slate-600'
                  }`}
              >
                <div className={`absolute top-1 w-5 h-5 bg-white rounded-full transition-transform ${useMcDropout ? 'translate-x-8' : 'translate-x-1'
                  }`} />
              </button>
            </div>

            {/* MC Samples Slider */}
            {useMcDropout && (
              <div className="animate-fade-in">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm text-slate-300">MC Samples</p>
                  <span className="badge badge-primary">{mcSamples}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="50"
                  step="5"
                  value={mcSamples}
                  onChange={(e) => setMcSamples(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>Faster (10)</span>
                  <span>More Accurate (50)</span>
                </div>
              </div>
            )}

            {/* Info Box */}
            <div className="mt-4 p-4 bg-purple-500/5 border border-purple-500/20 rounded-xl">
              <div className="flex items-start gap-3">
                <Info className="w-4 h-4 text-purple-400 mt-0.5" />
                <p className="text-sm text-slate-300">
                  MC Dropout runs multiple predictions to estimate how confident the model is.
                  Higher samples = more accurate uncertainty, but slower.
                </p>
              </div>
            </div>
          </div>

          {/* Analyze Button */}
          <button
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className="w-full btn-primary py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>{useMcDropout ? `Running ${mcSamples} predictions...` : 'Analyzing...'}</span>
              </>
            ) : (
              <>
                <Brain className="w-5 h-5" />
                <span>Analyze with AI</span>
              </>
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="glass-card bg-red-500/10 border-red-500/30 p-5 animate-fade-in">
              <div className="flex items-start gap-4">
                <div className="p-2 bg-red-500/20 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                </div>
                <div>
                  <p className="font-semibold text-red-300 mb-1">Analysis Failed</p>
                  <p className="text-sm text-red-200/80">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-semibold text-white">
                Analysis Results
              </h2>
              {savedToHistory && (
                <span className="flex items-center gap-1 px-2 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-full text-xs text-emerald-400">
                  <CheckCircle className="w-3 h-3" />
                  Saved
                </span>
              )}
            </div>

            {/* Download Report Button */}
            {result && selectedFile && (
              <button
                onClick={handleDownloadReport}
                disabled={reportLoading}
                className="btn-secondary py-2 px-4"
              >
                {reportLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    <span>Download PDF</span>
                  </>
                )}
              </button>
            )}
          </div>

          {/* Empty State */}
          {!result && !loading && (
            <div className="glass-card p-12 text-center">
              <div className="p-5 bg-white/5 border border-white/10 rounded-2xl w-fit mx-auto mb-5">
                <Brain className="w-12 h-12 text-slate-500" />
              </div>
              <h3 className="text-lg font-semibold text-slate-300 mb-2">
                Waiting for Analysis
              </h3>
              <p className="text-slate-500 max-w-sm mx-auto">
                Upload a histopathology image to receive AI-powered cancer detection with uncertainty estimation
              </p>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="glass-card bg-gradient-to-br from-emerald-500/5 to-cyan-500/5 p-12 text-center">
              <div className="p-5 bg-emerald-500/10 border border-emerald-500/30 rounded-2xl w-fit mx-auto mb-5">
                <Loader2 className="w-12 h-12 text-emerald-400 animate-spin" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">
                AI Analysis in Progress
              </h3>
              <p className="text-slate-400">
                {useMcDropout
                  ? `Running ${mcSamples} Monte Carlo samples for uncertainty estimation...`
                  : 'Analyzing tissue sample...'}
              </p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="space-y-6 animate-fade-in-up">
              {/* Diagnosis Card */}
              <div className={`glass-card overflow-hidden ${result.prediction === "Malignant"
                ? "bg-gradient-to-br from-red-500/10 to-rose-500/5 border-red-500/30"
                : "bg-gradient-to-br from-emerald-500/10 to-green-500/5 border-emerald-500/30"
                }`}>
                <div className="p-8">
                  <div className="flex items-center gap-4 mb-6">
                    <div className={`p-3 rounded-xl ${result.prediction === "Malignant"
                      ? "bg-red-500/20 border border-red-500/30"
                      : "bg-emerald-500/20 border border-emerald-500/30"
                      }`}>
                      <CheckCircle className={`w-7 h-7 ${result.prediction === "Malignant" ? "text-red-400" : "text-emerald-400"
                        }`} />
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                        Diagnosis Result
                      </p>
                      <p className={`text-4xl lg:text-5xl font-bold ${result.prediction === "Malignant" ? "text-red-400" : "text-emerald-400"
                        }`}>
                        {result.prediction}
                      </p>
                    </div>
                  </div>
                  <p className="text-slate-300">
                    {result.prediction === "Malignant"
                      ? "Cancerous tissue detected in the sample"
                      : "No cancer detected in the tissue sample"}
                  </p>
                </div>
              </div>

              {/* Confidence & Uncertainty Cards - Side by Side */}
              <div className="grid md:grid-cols-2 gap-4">
                {/* Confidence */}
                <div className="glass-card bg-gradient-to-br from-cyan-500/5 to-blue-500/5 p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-cyan-500/20 border border-cyan-500/30 rounded-lg">
                      <TrendingUp className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 uppercase tracking-wider">Confidence</p>
                      <p className="text-3xl font-bold text-cyan-400">{result.confidence}%</p>
                    </div>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-bar-fill bg-gradient-to-r from-cyan-500 to-blue-400"
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Uncertainty (if MC Dropout enabled) */}
                {result.uncertainty && (
                  <div className="glass-card bg-gradient-to-br from-purple-500/5 to-pink-500/5 p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 bg-purple-500/20 border border-purple-500/30 rounded-lg">
                        <Activity className="w-5 h-5 text-purple-400" />
                      </div>
                      <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wider">Uncertainty</p>
                        <p className="text-3xl font-bold text-purple-400">{result.uncertainty.score}%</p>
                      </div>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-bar-fill bg-gradient-to-r from-purple-500 to-pink-400"
                        style={{ width: `${result.uncertainty.score}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Reliability Assessment (MC Dropout) */}
              {result.reliability && result.clinical_recommendation && (
                <div className={`glass-card ${getReliabilityStyle(result.reliability).bg} ${getReliabilityStyle(result.reliability).border} p-6`}>
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl ${getReliabilityStyle(result.reliability).bg}`}>
                      {(() => {
                        const IconComponent = getReliabilityStyle(result.reliability).icon;
                        return <IconComponent className={`w-6 h-6 ${getReliabilityStyle(result.reliability).text}`} />;
                      })()}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className={`font-semibold ${getReliabilityStyle(result.reliability).text}`}>
                          {getReliabilityStyle(result.reliability).label}
                        </h4>
                        {result.n_samples && (
                          <span className="badge badge-primary text-xs">
                            {result.n_samples} samples
                          </span>
                        )}
                      </div>
                      <p className="text-slate-300 text-sm leading-relaxed">
                        {result.clinical_recommendation}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* PDF Report Card */}
              <div className="glass-card bg-gradient-to-br from-pink-500/5 to-rose-500/5 border-pink-500/20 p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-pink-500/20 border border-pink-500/30 rounded-xl">
                      <FileText className="w-6 h-6 text-pink-400" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-white">Download Analysis Report</h4>
                      <p className="text-sm text-slate-400">
                        Get a comprehensive PDF report with Grad-CAM visualization
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleDownloadReport}
                    disabled={reportLoading}
                    className="btn-primary py-3 px-5"
                  >
                    {reportLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Generating...</span>
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        <span>Download PDF</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Detailed Uncertainty Metrics */}
              {result.uncertainty && (
                <div className="glass-card p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    Uncertainty Metrics
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white/5 rounded-xl p-4">
                      <p className="text-xs text-slate-400 mb-1">Entropy</p>
                      <p className="text-lg font-semibold text-white">{result.uncertainty.entropy}</p>
                    </div>
                    <div className="bg-white/5 rounded-xl p-4">
                      <p className="text-xs text-slate-400 mb-1">Epistemic</p>
                      <p className="text-lg font-semibold text-white">{result.uncertainty.epistemic}</p>
                    </div>
                    <div className="bg-white/5 rounded-xl p-4">
                      <p className="text-xs text-slate-400 mb-1">Coefficient of Variation</p>
                      <p className="text-lg font-semibold text-white">{result.uncertainty.coefficient_of_variation}%</p>
                    </div>
                    <div className="bg-white/5 rounded-xl p-4">
                      <p className="text-xs text-slate-400 mb-1">Prediction Std</p>
                      <p className="text-lg font-semibold text-white">
                        ±{result.std?.benign || 0}% / ±{result.std?.malignant || 0}%
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Probability Distribution */}
              <div className="glass-card p-6">
                <h3 className="text-lg font-semibold text-white mb-6">
                  Probability Distribution
                </h3>

                <div className="space-y-5">
                  {/* Benign */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-emerald-400" />
                        <span className="font-medium text-white">Benign</span>
                        {result.std && (
                          <span className="text-xs text-slate-500">±{result.std.benign}%</span>
                        )}
                      </div>
                      <span className="text-xl font-bold text-emerald-400">
                        {result.probabilities.benign}%
                      </span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-bar-fill bg-gradient-to-r from-emerald-500 to-green-400"
                        style={{ width: `${result.probabilities.benign}%` }}
                      />
                    </div>
                  </div>

                  {/* Malignant */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-red-400" />
                        <span className="font-medium text-white">Malignant</span>
                        {result.std && (
                          <span className="text-xs text-slate-500">±{result.std.malignant}%</span>
                        )}
                      </div>
                      <span className="text-xl font-bold text-red-400">
                        {result.probabilities.malignant}%
                      </span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-bar-fill bg-gradient-to-r from-red-500 to-rose-400"
                        style={{ width: `${result.probabilities.malignant}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Model Info Footer */}
                <div className="mt-6 pt-4 border-t border-white/10">
                  <div className="flex flex-wrap items-center gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <Brain className="w-4 h-4 text-purple-400" />
                      <span className="text-slate-400">ResNet18</span>
                    </div>
                    {result.mc_dropout_enabled && (
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-cyan-400" />
                        <span className="text-slate-400">MC Dropout ({result.n_samples} samples)</span>
                      </div>
                    )}
                    <div className="flex items-center gap-2">
                      <Shield className="w-4 h-4 text-emerald-400" />
                      <span className="text-slate-400">92.86% Test Accuracy</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Predict;
