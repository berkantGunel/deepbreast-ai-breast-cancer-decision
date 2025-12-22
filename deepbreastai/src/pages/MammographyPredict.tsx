import { useState, useCallback } from "react";
import { predictMammography } from "../services/api";
import type { MammographyPredictionResponse } from "../services/api";
import { generatePDFReport } from "../utils/pdfReport";
import "./MammographyPredict.css";

function MammographyPredict() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [prediction, setPrediction] =
        useState<MammographyPredictionResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState<boolean>(false);

    const handleFileSelect = useCallback(
        (event: React.ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0];
            if (file) {
                setSelectedFile(file);
                setPreviewUrl(URL.createObjectURL(file));
                setPrediction(null);
                setError(null);
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
            if (file.type.startsWith("image/")) {
                setSelectedFile(file);
                setPreviewUrl(URL.createObjectURL(file));
                setPrediction(null);
                setError(null);
            }
        }
    }, []);

    const handlePredict = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);

        try {
            const result = await predictMammography(selectedFile, false);
            setPrediction(result);
        } catch (err: unknown) {
            const errorMessage =
                err instanceof Error ? err.message : "Prediction failed";
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    const getUrgencyColor = (urgency: string) => {
        switch (urgency) {
            case "low":
                return "#22c55e";
            case "medium":
                return "#f59e0b";
            case "high":
                return "#ef4444";
            default:
                return "#6b7280";
        }
    };

    const getPredictionColor = (pred: string) => {
        switch (pred) {
            case "Benign":
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

    return (
        <div className="mammography-predict">
            <div className="page-header">
                <h1>🩻 Mammography Analysis</h1>
                <p className="subtitle">
                    AI-powered BI-RADS classification for mammography images
                </p>
            </div>

            <div className="content-grid">
                {/* Upload Section */}
                <div className="upload-section">
                    <div className="card">
                        <h2>📤 Upload Mammogram</h2>

                        <div
                            className={`drop-zone ${dragActive ? "drag-active" : ""} ${previewUrl ? "has-image" : ""
                                }`}
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                        >
                            {previewUrl ? (
                                <div className="preview-container">
                                    <img
                                        src={previewUrl}
                                        alt="Preview"
                                        className="image-preview"
                                    />
                                    <button className="clear-btn" onClick={clearResults}>
                                        ✕
                                    </button>
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
                                            accept="image/*"
                                            onChange={handleFileSelect}
                                            hidden
                                        />
                                    </label>
                                </div>
                            )}
                        </div>

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
                        <h3>ℹ️ About BI-RADS Classification</h3>
                        <div className="birads-info">
                            <div className="birads-item benign">
                                <span className="birads-badge">BI-RADS 2-3</span>
                                <div>
                                    <strong>Benign</strong>
                                    <p>Low suspicion findings</p>
                                </div>
                            </div>
                            <div className="birads-item suspicious">
                                <span className="birads-badge">BI-RADS 4</span>
                                <div>
                                    <strong>Suspicious</strong>
                                    <p>Further evaluation needed</p>
                                </div>
                            </div>
                            <div className="birads-item malignant">
                                <span className="birads-badge">BI-RADS 5</span>
                                <div>
                                    <strong>Malignant</strong>
                                    <p>High suspicion of malignancy</p>
                                </div>
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
                            {/* Main Prediction Card */}
                            <div
                                className="card prediction-card"
                                style={{
                                    borderColor: getPredictionColor(prediction.prediction),
                                }}
                            >
                                <div className="prediction-header">
                                    <h2>Analysis Result</h2>
                                    <span className="birads-tag">
                                        {prediction.birads_category}
                                    </span>
                                </div>

                                <div className="prediction-main">
                                    <div
                                        className="prediction-label"
                                        style={{ color: getPredictionColor(prediction.prediction) }}
                                    >
                                        {prediction.prediction}
                                    </div>
                                    <div className="confidence-display">
                                        <span className="confidence-value">
                                            {prediction.confidence.toFixed(1)}%
                                        </span>
                                        <span className="confidence-label">Confidence</span>
                                    </div>
                                </div>

                                {/* Probability Bars */}
                                <div className="probability-section">
                                    <h4>Class Probabilities</h4>
                                    <div className="prob-bars">
                                        <div className="prob-row">
                                            <span className="prob-label">Benign</span>
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
                                            <span className="prob-label">Suspicious</span>
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
                                            <span className="prob-label">Malignant</span>
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
                                    borderColor: getUrgencyColor(
                                        prediction.recommendation.urgency
                                    ),
                                }}
                            >
                                <div className="rec-header">
                                    <h3>📋 Clinical Recommendation</h3>
                                    <span
                                        className="urgency-badge"
                                        style={{
                                            backgroundColor: getUrgencyColor(
                                                prediction.recommendation.urgency
                                            ),
                                        }}
                                    >
                                        {prediction.recommendation.urgency.toUpperCase()} URGENCY
                                    </span>
                                </div>

                                <div className="rec-action">
                                    <strong>{prediction.recommendation.action}</strong>
                                    <p>{prediction.recommendation.description}</p>
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

                            {/* PDF Download Button */}
                            <button
                                className="card"
                                onClick={handleDownloadPDF}
                                style={{
                                    width: "100%",
                                    padding: "1rem 1.5rem",
                                    background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
                                    border: "none",
                                    borderRadius: "12px",
                                    color: "white",
                                    fontSize: "1rem",
                                    fontWeight: 600,
                                    cursor: "pointer",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    gap: "0.5rem",
                                    transition: "transform 0.2s ease",
                                }}
                                onMouseOver={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
                                onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
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
                                    Upload a mammography image to receive AI-assisted BI-RADS
                                    classification and clinical recommendations.
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
