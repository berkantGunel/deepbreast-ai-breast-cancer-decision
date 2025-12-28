import { useState, useCallback } from "react";
import { Image as ImageIcon, Eye, PenTool, Activity, Layers, AlertTriangle, FileText, XCircle } from "lucide-react";
import { predictMammography, getMammographyPreview } from "../services/api";
import type { MammographyPredictionResponse } from "../services/api";
import { generatePDFReport } from "../utils/pdfReport";
import { useToast } from "../components/Toast";
import ImageViewer from "../components/ImageViewer";
import AnnotationCanvas from "../components/AnnotationCanvas";
import "./MammographyPredict.css";

function MammographyPredict() {
    const { showToast } = useToast();
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [prediction, setPrediction] =
        useState<MammographyPredictionResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(false);

    const [error, setError] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState<boolean>(false);
    const [viewMode, setViewMode] = useState<'simple' | 'zoom' | 'annotate'>('simple');
    const [activeTab, setActiveTab] = useState<'overview' | 'details' | 'features'>('overview');

    const processFile = async (file: File) => {
        setSelectedFile(file);
        setPrediction(null);
        setError(null);
        setPreviewUrl(null);

        const fileName = file.name.toLowerCase();
        const isTiff = fileName.endsWith('.tif') || fileName.endsWith('.tiff');
        const isDicom = fileName.endsWith('.dcm') || fileName.endsWith('.dicom');
        const needsServerPreview = isTiff || isDicom;

        if (needsServerPreview) {
            try {
                const response = await getMammographyPreview(file);
                if (response.success && response.image) {
                    setPreviewUrl(response.image);
                } else {
                    throw new Error("Failed to generate preview");
                }
            } catch (err) {
                console.error("Preview error:", err);
                showToast("Could not generate preview for this file format, but you can still run analysis.", "warning");
                // Set a placeholder or generic icon if preview fails
            }
        } else {
            // Standard image formats
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    const handleFileSelect = useCallback(
        (event: React.ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0];
            if (file) {
                processFile(file);
            }
        },
        []
    );

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            const fileName = file.name.toLowerCase();
            if (file.type.startsWith("image/") || fileName.endsWith('.tif') || fileName.endsWith('.tiff') || fileName.endsWith('.dcm') || fileName.endsWith('.dicom')) {
                processFile(file);
            }
        }
    }, []);

    const handlePredict = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);

        try {
            const result = await predictMammography(selectedFile, "en");
            setPrediction(result);

            const pathologyName = result.pathology_classification?.name || result.prediction;
            const confidence = (result.pathology_classification?.confidence * 100 || result.confidence).toFixed(1);
            const riskLevel = result.risk_assessment?.level || '';

            showToast(
                `Analysis Complete: ${pathologyName} - ${riskLevel} Risk (${confidence}% confidence)`,
                result.pathology_classification?.is_malignant ? 'warning' : 'success'
            );
        } catch (err: unknown) {
            const errorMessage =
                err instanceof Error ? err.message : "Prediction failed";
            setError(errorMessage);
            showToast(errorMessage, 'error');
        } finally {
            setLoading(false);
        }
    };

    const getRiskColor = (level: string) => {
        switch (level) {
            case "Low":
                return "#22c55e";
            case "Low-Medium":
                return "#84cc16";
            case "Medium":
                return "#f59e0b";
            case "Medium-High":
                return "#f97316";
            case "High":
                return "#ef4444";
            case "Critical":
                return "#dc2626";
            default:
                return "#6b7280";
        }
    };

    const getPredictionColor = (pred: string) => {
        switch (pred) {
            case "Benign":
            case "Normal":
                return "#22c55e";
            case "Suspicious":
                return "#f59e0b";
            case "Malignant":
                return "#ef4444";
            default:
                return "#6b7280";
        }
    };

    const clearResults = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setPrediction(null);
        setError(null);
    };

    const handleDownloadPDF = async () => {
        if (!prediction) return;

        try {
            await generatePDFReport({
                type: "mammography",
                prediction: prediction.prediction,
                confidence: prediction.confidence,
                probabilities: prediction.probabilities,
                biradsCategory: prediction.birads_category,
                recommendation: prediction.recommendation,
                imageUrl: previewUrl || undefined,
                timestamp: new Date().toISOString(),
            });
        } catch (err) {
            console.error("PDF generation failed:", err);
        }
    };

    const formatFeatureValue = (value: number): string => {
        if (value === 0) return "0";
        if (Math.abs(value) < 0.001) return value.toExponential(2);
        if (Math.abs(value) > 1000) return value.toFixed(0);
        return value.toFixed(4);
    };

    return (
        <div className="mammography-predict">
            <div className="page-header">
                <h1>🩻 Mammography Analysis</h1>
                <p className="subtitle">
                    Classical ML-based detailed mammography image analysis
                </p>
                <div className="model-badge">
                    <span className="badge-icon">🔬</span>
                    <span>DMID Dataset • 81% Accuracy • 78 Features</span>
                </div>
            </div>

            <div className="content-grid">
                {/* Upload Section */}
                <div className="upload-section">
                    <div className="card">
                        <h2>📤 Upload Mammogram</h2>

                        <div
                            className={`drop-zone ${dragActive ? "drag-active" : ""} ${previewUrl ? "has-image" : ""}`}
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                        >
                            {selectedFile ? (
                                <div className="preview-container" style={{ minHeight: 'auto', padding: '0', border: 'none', background: 'transparent' }}>
                                    <div style={{
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "space-between",
                                        width: "100%",
                                        backgroundColor: "rgba(30, 41, 59, 0.4)",
                                        padding: "1rem",
                                        borderRadius: "8px",
                                        border: "1px solid rgba(148, 163, 184, 0.1)"
                                    }}>
                                        <h3 style={{
                                            color: "#f1f5f9",
                                            fontWeight: 600,
                                            margin: 0,
                                            fontSize: "1rem"
                                        }}>Selected Image</h3>

                                        <button
                                            onClick={clearResults}
                                            style={{
                                                background: "none",
                                                border: "none",
                                                cursor: "pointer",
                                                padding: "4px",
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                                color: "#94a3b8",
                                                transition: "color 0.2s"
                                            }}
                                            onMouseEnter={(e) => e.currentTarget.style.color = "#ef4444"}
                                            onMouseLeave={(e) => e.currentTarget.style.color = "#94a3b8"}
                                            title="Remove file"
                                        >
                                            <XCircle size={20} />
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="drop-content">
                                    <span className="drop-icon">🩻</span>
                                    <p>Drag & drop mammography image here</p>
                                    <span className="or">or</span>
                                    <label className="browse-btn">
                                        Browse Files
                                        <input
                                            type="file"
                                            accept="image/*,.tif,.tiff,.dcm,.dicom"
                                            onChange={handleFileSelect}
                                            hidden
                                        />
                                    </label>
                                    <span className="file-types">Supports JPEG, PNG, TIFF, DICOM</span>
                                </div>
                            )}
                        </div>

                        {/* View Mode Toggle - Only show when image is loaded */}
                        {previewUrl && (
                            <>
                                <div className="view-mode-toggle">
                                    <button
                                        className={`view-mode-btn ${viewMode === 'simple' ? 'active' : ''}`}
                                        onClick={() => setViewMode('simple')}
                                    >
                                        <ImageIcon size={16} />
                                        Simple
                                    </button>
                                    <button
                                        className={`view-mode-btn ${viewMode === 'zoom' ? 'active zoom' : ''}`}
                                        onClick={() => setViewMode('zoom')}
                                    >
                                        <Eye size={16} />
                                        Zoom/Pan
                                    </button>
                                    <button
                                        className={`view-mode-btn ${viewMode === 'annotate' ? 'active annotate' : ''}`}
                                        onClick={() => setViewMode('annotate')}
                                    >
                                        <PenTool size={16} />
                                        Annotate
                                    </button>
                                </div>

                                {/* Image Display based on View Mode */}
                                <div className="image-viewer-wrapper">
                                    {viewMode === 'simple' && (
                                        <img
                                            src={previewUrl}
                                            alt="Preview"
                                            className="image-preview"
                                        />
                                    )}
                                    {viewMode === 'zoom' && (
                                        <ImageViewer src={previewUrl} alt="Mammography Preview" />
                                    )}
                                    {viewMode === 'annotate' && (
                                        <AnnotationCanvas imageSrc={previewUrl} />
                                    )}
                                </div>
                            </>
                        )}

                        {selectedFile && (
                            <div className="file-info">
                                <span className="file-name">{selectedFile.name}</span>
                                <span className="file-size">
                                    {(selectedFile.size / 1024).toFixed(1)} KB
                                </span>
                            </div>
                        )}

                        <button
                            className="analyze-btn"
                            onClick={handlePredict}
                            disabled={!selectedFile || loading}
                        >
                            {loading ? (
                                <>
                                    <span className="spinner"></span>
                                    Analyzing...
                                </>
                            ) : (
                                <>🔍 Analyze Mammogram</>
                            )}
                        </button>
                    </div>

                    {/* Info Card */}
                    <div className="card info-card">
                        <h3>ℹ️ Model Information</h3>
                        <div className="model-info-grid">
                            <div className="info-item">
                                <span className="info-label">Dataset</span>
                                <span className="info-value">DMID Mammography</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Features</span>
                                <span className="info-value">78</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Model</span>
                                <span className="info-value">RF + GB Ensemble</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Accuracy</span>
                                <span className="info-value success">81.37%</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Results Section */}
                <div className="results-section">
                    {error && (
                        <div className="card error-card">
                            <h3>❌ Error</h3>
                            <p>{error}</p>
                        </div>
                    )}

                    {prediction && (
                        <>
                            {/* Tab Navigation */}
                            <div className="result-tabs">
                                <button
                                    className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('overview')}
                                >
                                    <Activity size={16} />
                                    Overview
                                </button>
                                <button
                                    className={`tab-btn ${activeTab === 'details' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('details')}
                                >
                                    <Layers size={16} />
                                    Detailed Analysis
                                </button>
                                <button
                                    className={`tab-btn ${activeTab === 'features' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('features')}
                                >
                                    <FileText size={16} />
                                    Features
                                </button>
                            </div>

                            {/* Overview Tab */}
                            {activeTab === 'overview' && (
                                <>
                                    {/* Main Prediction Card */}
                                    <div
                                        className="card prediction-card"
                                        style={{
                                            borderColor: getPredictionColor(prediction.prediction),
                                        }}
                                    >
                                        <div className="prediction-header">
                                            <h2>Pathology Result</h2>
                                            <span
                                                className="risk-badge"
                                                style={{ backgroundColor: getRiskColor(prediction.risk_assessment?.level || '') }}
                                            >
                                                {prediction.risk_assessment?.level} Risk
                                            </span>
                                        </div>

                                        <div className="prediction-main">
                                            <div
                                                className="prediction-label"
                                                style={{ color: getPredictionColor(prediction.prediction) }}
                                            >
                                                {prediction.pathology_classification?.name || prediction.prediction}
                                            </div>
                                            <div className="confidence-display">
                                                <span className="confidence-value">
                                                    {(prediction.pathology_classification?.confidence * 100 || prediction.confidence).toFixed(1)}%
                                                </span>
                                                <span className="confidence-label">Confidence</span>
                                            </div>
                                        </div>

                                        {/* Probability Bars */}
                                        <div className="probability-section">
                                            <h4>Class Probabilities</h4>
                                            <div className="prob-bars">
                                                <div className="prob-row">
                                                    <span className="prob-label">Benign (B)</span>
                                                    <div className="prob-bar-container">
                                                        <div
                                                            className="prob-bar benign"
                                                            style={{
                                                                width: `${prediction.probabilities.benign}%`,
                                                            }}
                                                        ></div>
                                                    </div>
                                                    <span className="prob-value">
                                                        {prediction.probabilities.benign.toFixed(1)}%
                                                    </span>
                                                </div>
                                                <div className="prob-row">
                                                    <span className="prob-label">Normal (N)</span>
                                                    <div className="prob-bar-container">
                                                        <div
                                                            className="prob-bar suspicious"
                                                            style={{
                                                                width: `${prediction.probabilities.suspicious}%`,
                                                            }}
                                                        ></div>
                                                    </div>
                                                    <span className="prob-value">
                                                        {prediction.probabilities.suspicious.toFixed(1)}%
                                                    </span>
                                                </div>
                                                <div className="prob-row">
                                                    <span className="prob-label">Malignant (M)</span>
                                                    <div className="prob-bar-container">
                                                        <div
                                                            className="prob-bar malignant"
                                                            style={{
                                                                width: `${prediction.probabilities.malignant}%`,
                                                            }}
                                                        ></div>
                                                    </div>
                                                    <span className="prob-value">
                                                        {prediction.probabilities.malignant.toFixed(1)}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Recommendation Card */}
                                    <div
                                        className="card recommendation-card"
                                        style={{
                                            borderColor: getRiskColor(prediction.risk_assessment?.level || ''),
                                        }}
                                    >
                                        <div className="rec-header">
                                            <h3>📋 Clinical Recommendation</h3>
                                            <span
                                                className="urgency-badge"
                                                style={{
                                                    backgroundColor: getRiskColor(prediction.risk_assessment?.level || ''),
                                                }}
                                            >
                                                <AlertTriangle size={14} />
                                                {prediction.risk_assessment?.level}
                                            </span>
                                        </div>

                                        <div className="rec-action">
                                            <strong>{prediction.risk_assessment?.recommendation}</strong>
                                        </div>

                                        <div className="rec-steps">
                                            <h4>Recommended Next Steps:</h4>
                                            <ul>
                                                {prediction.recommendation.next_steps.map((step, i) => (
                                                    <li key={i}>{step}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                </>
                            )}

                            {/* Details Tab */}
                            {activeTab === 'details' && (
                                <>
                                    {/* Tissue Analysis */}
                                    <div className="card detail-card">
                                        <h3>🔬 Tissue Analysis</h3>
                                        <div className="detail-content">
                                            <div className="detail-main">
                                                <span className="detail-type">{prediction.tissue_analysis?.type}</span>
                                                <span className="detail-name">
                                                    {prediction.tissue_analysis?.name}
                                                </span>
                                                <span className="detail-confidence">
                                                    {((prediction.tissue_analysis?.confidence || 0) * 100).toFixed(1)}% confidence
                                                </span>
                                            </div>
                                            <div className="detail-probs">
                                                {prediction.tissue_analysis?.probabilities && Object.entries(prediction.tissue_analysis.probabilities).map(([key, value]) => (
                                                    <div key={key} className="mini-prob">
                                                        <span>{key}</span>
                                                        <div className="mini-bar">
                                                            <div style={{ width: `${(value as number) * 100}%` }}></div>
                                                        </div>
                                                        <span>{((value as number) * 100).toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Abnormality Detection */}
                                    <div className="card detail-card">
                                        <h3>🎯 Abnormality Detection</h3>
                                        <div className="detail-content">
                                            <div className="detail-main">
                                                <span className="detail-type">{prediction.abnormality_detection?.type}</span>
                                                <span className="detail-name">
                                                    {prediction.abnormality_detection?.name}
                                                </span>
                                                <span className="detail-confidence">
                                                    {((prediction.abnormality_detection?.confidence || 0) * 100).toFixed(1)}% confidence
                                                </span>
                                            </div>
                                            <div className="detail-probs">
                                                {prediction.abnormality_detection?.probabilities && Object.entries(prediction.abnormality_detection.probabilities).map(([key, value]) => (
                                                    <div key={key} className="mini-prob">
                                                        <span>{key}</span>
                                                        <div className="mini-bar">
                                                            <div style={{ width: `${(value as number) * 100}%` }}></div>
                                                        </div>
                                                        <span>{((value as number) * 100).toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Pathology Details */}
                                    <div className="card detail-card">
                                        <h3>🩺 Pathology Details</h3>
                                        <div className="detail-content">
                                            <div className="detail-main">
                                                <span className={`detail-type ${prediction.pathology_classification?.is_malignant ? 'danger' : 'success'}`}>
                                                    {prediction.pathology_classification?.class}
                                                </span>
                                                <span className="detail-name">
                                                    {prediction.pathology_classification?.name}
                                                </span>
                                                <span className="detail-confidence">
                                                    {((prediction.pathology_classification?.confidence || 0) * 100).toFixed(1)}% confidence
                                                </span>
                                            </div>
                                            {prediction.pathology_classification?.is_malignant && (
                                                <div className="malignant-warning">
                                                    <AlertTriangle size={20} />
                                                    <span>Malignant finding detected - Urgent evaluation recommended</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </>
                            )}

                            {/* Features Tab */}
                            {activeTab === 'features' && (
                                <>
                                    {/* Texture Features */}
                                    <div className="card feature-card">
                                        <h3>📊 Texture Features (GLCM)</h3>
                                        <div className="feature-grid">
                                            {prediction.feature_analysis?.texture && Object.entries(prediction.feature_analysis.texture).map(([key, value]) => (
                                                <div key={key} className="feature-item">
                                                    <span className="feature-name">{key}</span>
                                                    <span className="feature-value">{formatFeatureValue(value as number)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Morphological Features */}
                                    <div className="card feature-card">
                                        <h3>📐 Morphological Features</h3>
                                        <div className="feature-grid">
                                            {prediction.feature_analysis?.morphology && Object.entries(prediction.feature_analysis.morphology).map(([key, value]) => (
                                                <div key={key} className="feature-item">
                                                    <span className="feature-name">{key}</span>
                                                    <span className="feature-value">{formatFeatureValue(value as number)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Model Info */}
                                    <div className="card feature-card">
                                        <h3>🤖 Model Information</h3>
                                        <div className="feature-grid">
                                            <div className="feature-item">
                                                <span className="feature-name">Type</span>
                                                <span className="feature-value">{prediction.model_info?.type}</span>
                                            </div>
                                            <div className="feature-item">
                                                <span className="feature-name">Features</span>
                                                <span className="feature-value">{prediction.model_info?.features}</span>
                                            </div>
                                            <div className="feature-item">
                                                <span className="feature-name">Dataset</span>
                                                <span className="feature-value">{prediction.model_info?.dataset}</span>
                                            </div>
                                            <div className="feature-item">
                                                <span className="feature-name">Version</span>
                                                <span className="feature-value">{prediction.model_info?.version}</span>
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}

                            {/* PDF Download Button */}
                            <button
                                className="card pdf-download-btn"
                                onClick={handleDownloadPDF}
                            >
                                📄 Download PDF Report
                            </button>

                            {/* Disclaimer */}
                            <div className="card disclaimer-card">
                                <p>
                                    ⚠️ <strong>Important:</strong> This AI analysis is intended to
                                    assist healthcare professionals and should not be used as the
                                    sole basis for clinical decisions. All findings must be
                                    reviewed and confirmed by qualified radiologists.
                                </p>
                            </div>
                        </>
                    )}

                    {!prediction && !error && !loading && (
                        <div className="card placeholder-card">
                            <div className="placeholder-content">
                                <span className="placeholder-icon">🔬</span>
                                <h3>Ready for Analysis</h3>
                                <p>
                                    Upload a mammography image to receive AI-assisted analysis
                                    including tissue type, abnormality detection, and pathology classification.
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default MammographyPredict;
