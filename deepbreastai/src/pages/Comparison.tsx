import { useState, useCallback } from "react";
import { GitCompare, Upload, Calendar, X, ArrowLeftRight, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

interface ComparisonImage {
    id: string;
    file: File;
    url: string;
    date: string;
    label: string;
}

const Comparison = () => {
    const [leftImage, setLeftImage] = useState<ComparisonImage | null>(null);
    const [rightImage, setRightImage] = useState<ComparisonImage | null>(null);
    const [zoomLevel, setZoomLevel] = useState(100);
    const [syncZoom, setSyncZoom] = useState(true);

    const handleImageUpload = useCallback((side: "left" | "right") => {
        return (event: React.ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0];
            if (file) {
                const newImage: ComparisonImage = {
                    id: `${side}-${Date.now()}`,
                    file,
                    url: URL.createObjectURL(file),
                    date: new Date().toISOString(),
                    label: file.name.replace(/\.[^/.]+$/, ""),
                };

                if (side === "left") {
                    if (leftImage) URL.revokeObjectURL(leftImage.url);
                    setLeftImage(newImage);
                } else {
                    if (rightImage) URL.revokeObjectURL(rightImage.url);
                    setRightImage(newImage);
                }
            }
        };
    }, [leftImage, rightImage]);

    const handleDrop = useCallback((side: "left" | "right") => {
        return (event: React.DragEvent) => {
            event.preventDefault();
            const file = event.dataTransfer.files?.[0];
            if (file && file.type.startsWith("image/")) {
                const newImage: ComparisonImage = {
                    id: `${side}-${Date.now()}`,
                    file,
                    url: URL.createObjectURL(file),
                    date: new Date().toISOString(),
                    label: file.name.replace(/\.[^/.]+$/, ""),
                };

                if (side === "left") {
                    if (leftImage) URL.revokeObjectURL(leftImage.url);
                    setLeftImage(newImage);
                } else {
                    if (rightImage) URL.revokeObjectURL(rightImage.url);
                    setRightImage(newImage);
                }
            }
        };
    }, [leftImage, rightImage]);

    const clearImage = (side: "left" | "right") => {
        if (side === "left" && leftImage) {
            URL.revokeObjectURL(leftImage.url);
            setLeftImage(null);
        } else if (side === "right" && rightImage) {
            URL.revokeObjectURL(rightImage.url);
            setRightImage(null);
        }
    };

    const swapImages = () => {
        const temp = leftImage;
        setLeftImage(rightImage);
        setRightImage(temp);
    };

    const handleZoom = (delta: number) => {
        setZoomLevel(prev => Math.max(50, Math.min(200, prev + delta)));
    };

    const resetZoom = () => setZoomLevel(100);

    const renderDropZone = (side: "left" | "right", image: ComparisonImage | null) => (
        <div className="sharp-card" style={{ flex: 1, padding: "1.5rem", minHeight: "400px", display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h3 style={{ color: "#f1f5f9", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <Calendar style={{ width: "18px", height: "18px", color: side === "left" ? "#8b5cf6" : "#10b981" }} />
                    {side === "left" ? "Previous Image" : "Current Image"}
                </h3>
                {image && (
                    <button
                        onClick={() => clearImage(side)}
                        style={{
                            padding: "0.5rem",
                            background: "rgba(239, 68, 68, 0.15)",
                            border: "1px solid rgba(239, 68, 68, 0.3)",
                            borderRadius: "8px",
                            color: "#f87171",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center"
                        }}
                    >
                        <X style={{ width: "16px", height: "16px" }} />
                    </button>
                )}
            </div>

            {image ? (
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1rem" }}>
                    <div style={{
                        flex: 1,
                        background: "rgba(0,0,0,0.3)",
                        borderRadius: "12px",
                        overflow: "hidden",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        position: "relative"
                    }}>
                        <img
                            src={image.url}
                            alt={image.label}
                            style={{
                                maxWidth: "100%",
                                maxHeight: "100%",
                                objectFit: "contain",
                                transform: `scale(${zoomLevel / 100})`,
                                transition: "transform 0.2s ease"
                            }}
                        />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ color: "#94a3b8", fontSize: "0.875rem" }}>
                            {image.label}
                        </span>
                        <span style={{ color: "#64748b", fontSize: "0.75rem" }}>
                            {new Date(image.date).toLocaleDateString()}
                        </span>
                    </div>
                </div>
            ) : (
                <div
                    onDrop={handleDrop(side)}
                    onDragOver={(e) => e.preventDefault()}
                    style={{
                        flex: 1,
                        border: "2px dashed rgba(139, 92, 246, 0.3)",
                        borderRadius: "12px",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: "1rem",
                        background: "rgba(139, 92, 246, 0.05)",
                        cursor: "pointer",
                        transition: "all 0.3s ease"
                    }}
                >
                    <Upload style={{ width: "48px", height: "48px", color: "#8b5cf6", opacity: 0.5 }} />
                    <div style={{ textAlign: "center" }}>
                        <p style={{ color: "#94a3b8", marginBottom: "0.5rem" }}>Drag & drop an image here</p>
                        <span style={{ color: "#64748b", fontSize: "0.875rem" }}>or</span>
                    </div>
                    <label style={{
                        padding: "0.75rem 1.5rem",
                        background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
                        borderRadius: "10px",
                        color: "white",
                        fontWeight: 600,
                        cursor: "pointer",
                        transition: "transform 0.2s ease"
                    }}>
                        Browse Files
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageUpload(side)}
                            style={{ display: "none" }}
                        />
                    </label>
                </div>
            )}
        </div>
    );

    return (
        <div className="sharp-page">
            {/* Header */}
            <section className="sharp-section">
                <div className="sharp-header">
                    <h1 style={{ fontSize: "2rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
                        <GitCompare style={{ width: "32px", height: "32px", color: "#8b5cf6" }} />
                        Image Comparison
                    </h1>
                    <p className="subtitle">Compare mammography images side by side to track changes over time</p>
                </div>
            </section>

            {/* Zoom Controls */}
            <section className="sharp-section">
                <div className="sharp-card" style={{
                    padding: "1rem 1.5rem",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: "1rem"
                }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                        <span style={{ color: "#94a3b8", fontSize: "0.875rem" }}>Zoom: {zoomLevel}%</span>
                        <div style={{ display: "flex", gap: "0.5rem" }}>
                            <button
                                onClick={() => handleZoom(-10)}
                                className="sharp-btn-secondary"
                                style={{ padding: "0.5rem" }}
                            >
                                <ZoomOut style={{ width: "18px", height: "18px" }} />
                            </button>
                            <button
                                onClick={() => handleZoom(10)}
                                className="sharp-btn-secondary"
                                style={{ padding: "0.5rem" }}
                            >
                                <ZoomIn style={{ width: "18px", height: "18px" }} />
                            </button>
                            <button
                                onClick={resetZoom}
                                className="sharp-btn-secondary"
                                style={{ padding: "0.5rem" }}
                            >
                                <RotateCcw style={{ width: "18px", height: "18px" }} />
                            </button>
                        </div>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
                            <input
                                type="checkbox"
                                checked={syncZoom}
                                onChange={(e) => setSyncZoom(e.target.checked)}
                                style={{ accentColor: "#8b5cf6" }}
                            />
                            <span style={{ color: "#94a3b8", fontSize: "0.875rem" }}>Sync Zoom</span>
                        </label>

                        {leftImage && rightImage && (
                            <button
                                onClick={swapImages}
                                className="sharp-btn-primary"
                                style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                            >
                                <ArrowLeftRight style={{ width: "18px", height: "18px" }} />
                                Swap Images
                            </button>
                        )}
                    </div>
                </div>
            </section>

            {/* Comparison View */}
            <section className="sharp-section">
                <div style={{ display: "flex", gap: "1.5rem" }} className="comparison-grid">
                    {renderDropZone("left", leftImage)}

                    {/* Divider */}
                    <div style={{
                        width: "2px",
                        background: "linear-gradient(180deg, transparent, #8b5cf6, transparent)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        position: "relative"
                    }}>
                        <div style={{
                            position: "absolute",
                            width: "40px",
                            height: "40px",
                            background: "#1e293b",
                            borderRadius: "50%",
                            border: "2px solid #8b5cf6",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center"
                        }}>
                            <ArrowLeftRight style={{ width: "18px", height: "18px", color: "#8b5cf6" }} />
                        </div>
                    </div>

                    {renderDropZone("right", rightImage)}
                </div>
            </section>

            {/* Info Card */}
            <section className="sharp-section">
                <div className="sharp-card" style={{
                    padding: "1.5rem",
                    background: "linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.05))",
                    borderColor: "rgba(139, 92, 246, 0.2)"
                }}>
                    <h4 style={{ color: "#a78bfa", fontWeight: 600, marginBottom: "0.75rem" }}>
                        💡 How to Use
                    </h4>
                    <ul style={{ color: "#c4b5fd", fontSize: "0.9rem", lineHeight: 1.8, margin: 0, paddingLeft: "1.5rem" }}>
                        <li>Upload a <strong>previous</strong> mammography image on the left side</li>
                        <li>Upload the <strong>current</strong> mammography image on the right side</li>
                        <li>Use zoom controls to examine details in both images</li>
                        <li>Compare changes in tissue density, calcifications, or masses over time</li>
                        <li>Click "Swap Images" to switch the positions</li>
                    </ul>
                </div>
            </section>

            <style>{`
        @media (max-width: 768px) {
          .comparison-grid {
            flex-direction: column !important;
          }
          .comparison-grid > div:nth-child(2) {
            display: none !important;
          }
        }
      `}</style>
        </div>
    );
};

export default Comparison;
