import { useState, useEffect } from "react";
import {
    Clock,
    Search,
    Trash2,
    Filter,
    CheckCircle,
    AlertCircle,
    TrendingUp,
    Activity,
    FileImage,
    Upload,
    Loader2,
    RefreshCw,
    BarChart3,
    X,
    ChevronLeft,
    ChevronRight,
    Brain,
    ScanLine,
    Info,
} from "lucide-react";
import {
    getHistory,
    getHistoryStats,
    deleteHistoryRecord,
    batchUpload,
    type HistoryRecord,
    type HistoryStats,
    type BatchResponse,
} from "../services/api";

type ModelTab = "histopathology" | "mammography";

const History = () => {
    const [activeTab, setActiveTab] = useState<ModelTab>("histopathology");
    const [records, setRecords] = useState<HistoryRecord[]>([]);
    const [stats, setStats] = useState<HistoryStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState("");
    const [predictionFilter, setPredictionFilter] = useState<"Benign" | "Malignant" | "">("");
    const [currentPage, setCurrentPage] = useState(0);
    const [totalCount, setTotalCount] = useState(0);
    const pageSize = 20;
    const [batchFiles, setBatchFiles] = useState<File[]>([]);
    const [batchLoading, setBatchLoading] = useState(false);
    const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);

    const loadHistory = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await getHistory({
                limit: pageSize,
                offset: currentPage * pageSize,
                prediction: predictionFilter || undefined,
                search: searchTerm || undefined,
            });
            setRecords(response.records);
            setTotalCount(response.count);
        } catch (err) {
            setError("Failed to load history");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const loadStats = async () => {
        try {
            const response = await getHistoryStats();
            setStats(response.statistics);
        } catch (err) {
            console.error("Failed to load stats:", err);
        }
    };

    useEffect(() => {
        if (activeTab === "histopathology") {
            loadHistory();
            loadStats();
        }
    }, [currentPage, predictionFilter, activeTab]);

    useEffect(() => {
        if (activeTab !== "histopathology") return;
        const timeout = setTimeout(() => {
            setCurrentPage(0);
            loadHistory();
        }, 300);
        return () => clearTimeout(timeout);
    }, [searchTerm]);

    const handleDelete = async (id: number) => {
        if (!confirm("Are you sure you want to delete this record?")) return;
        try {
            await deleteHistoryRecord(id);
            loadHistory();
            loadStats();
        } catch (err) {
            console.error("Delete failed:", err);
        }
    };

    const handleBatchUpload = async () => {
        if (batchFiles.length === 0) return;
        setBatchLoading(true);
        setBatchResult(null);
        try {
            const result = await batchUpload(batchFiles, { useMcDropout: true, mcSamples: 20 });
            setBatchResult(result);
            setBatchFiles([]);
            loadHistory();
            loadStats();
        } catch (err) {
            setError("Batch upload failed");
            console.error(err);
        } finally {
            setBatchLoading(false);
        }
    };

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    return (
        <div className="sharp-page">
            {/* Header */}
            <div className="sharp-header">
                <h1>Analysis History</h1>
                <p className="subtitle">View past analyses and upload multiple images</p>
            </div>

            {/* Tabs */}
            <div className="sharp-tabs">
                <button className={`sharp-tab ${activeTab === "histopathology" ? "active" : ""}`} onClick={() => setActiveTab("histopathology")}>
                    <Brain style={{ width: "18px", height: "18px" }} /> Histopathology
                </button>
                <button className={`sharp-tab ${activeTab === "mammography" ? "active" : ""}`} onClick={() => setActiveTab("mammography")}>
                    <ScanLine style={{ width: "18px", height: "18px" }} /> Mammography
                </button>
            </div>

            {/* Stats Cards - Histopathology */}
            {activeTab === "histopathology" && stats && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "2rem" }} className="stats-grid-history">
                    {[
                        { label: "Total Analyses", value: stats.total_analyses, color: "#8b5cf6", icon: BarChart3 },
                        { label: "Benign", value: stats.by_prediction?.Benign || 0, color: "#22c55e", icon: CheckCircle },
                        { label: "Malignant", value: stats.by_prediction?.Malignant || 0, color: "#ef4444", icon: AlertCircle },
                        { label: "Avg Confidence", value: `${stats.average_confidence}%`, color: "#06b6d4", icon: TrendingUp },
                    ].map((stat) => (
                        <div key={stat.label} className="sharp-metric-card">
                            <stat.icon style={{ width: "24px", height: "24px", color: stat.color, marginBottom: "0.75rem" }} />
                            <div style={{ fontSize: "2rem", fontWeight: 700, color: stat.color }}>{stat.value}</div>
                            <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>{stat.label}</div>
                        </div>
                    ))}
                </div>
            )}

            {/* Histopathology History */}
            {activeTab === "histopathology" && (
                <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "2rem" }} className="history-grid">
                    {/* History List */}
                    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                        {/* Search & Filter */}
                        <div className="sharp-card" style={{ padding: "1rem" }}>
                            <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                                <div style={{ flex: 1, position: "relative", minWidth: "200px" }}>
                                    <Search style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", width: "18px", height: "18px", color: "#64748b" }} />
                                    <input
                                        type="text"
                                        placeholder="Search files..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        style={{ width: "100%", paddingLeft: "40px", padding: "0.75rem 1rem 0.75rem 40px", background: "rgba(30, 41, 59, 0.6)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", color: "#f1f5f9", fontSize: "0.9rem" }}
                                    />
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                                    <Filter style={{ width: "18px", height: "18px", color: "#64748b" }} />
                                    <select value={predictionFilter} onChange={(e) => setPredictionFilter(e.target.value as "" | "Benign" | "Malignant")} style={{ padding: "0.75rem 1rem", background: "rgba(30, 41, 59, 0.6)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", color: "#f1f5f9", fontSize: "0.9rem" }}>
                                        <option value="">All Results</option>
                                        <option value="Benign">Benign Only</option>
                                        <option value="Malignant">Malignant Only</option>
                                    </select>
                                    <button onClick={() => { loadHistory(); loadStats(); }} style={{ padding: "0.75rem", background: "rgba(30, 41, 59, 0.6)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", cursor: "pointer" }}>
                                        <RefreshCw style={{ width: "18px", height: "18px", color: "#94a3b8", ...(loading ? { animation: "spin 1s linear infinite" } : {}) }} />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Records */}
                        {loading && records.length === 0 ? (
                            <div className="sharp-card" style={{ textAlign: "center", padding: "4rem 2rem" }}>
                                <Loader2 style={{ width: "40px", height: "40px", color: "#8b5cf6", animation: "spin 1s linear infinite", margin: "0 auto 1rem" }} />
                                <p style={{ color: "#94a3b8" }}>Loading history...</p>
                            </div>
                        ) : records.length === 0 ? (
                            <div className="sharp-card sharp-empty-state">
                                <div className="icon-wrapper"><Clock style={{ width: "32px", height: "32px", color: "#8b5cf6" }} /></div>
                                <h3>No Records Found</h3>
                                <p>Analyze some images to see them here</p>
                            </div>
                        ) : (
                            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                                {records.map((record) => (
                                    <div key={record.id} className="sharp-card sharp-list-item" style={{ borderLeftColor: record.prediction === "Malignant" ? "#ef4444" : "#22c55e", padding: "1rem" }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: "1rem", width: "100%" }}>
                                            {record.thumbnail ? (
                                                <img src={`data:image/jpeg;base64,${record.thumbnail}`} alt={record.filename} style={{ width: "56px", height: "56px", borderRadius: "10px", objectFit: "cover", flexShrink: 0 }} />
                                            ) : (
                                                <div style={{ width: "56px", height: "56px", background: "rgba(255,255,255,0.05)", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                                                    <FileImage style={{ width: "24px", height: "24px", color: "#64748b" }} />
                                                </div>
                                            )}
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.25rem" }}>
                                                    <h4 style={{ color: "#f1f5f9", fontWeight: 500, fontSize: "0.9rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{record.filename}</h4>
                                                    {record.is_batch && <span className="sharp-badge primary" style={{ fontSize: "0.7rem" }}>Batch</span>}
                                                </div>
                                                <p style={{ color: "#64748b", fontSize: "0.8rem" }}>{formatDate(record.created_at)}</p>
                                            </div>
                                            <div style={{ textAlign: "right", flexShrink: 0 }}>
                                                <p style={{ fontWeight: 600, fontSize: "0.9rem", color: record.prediction === "Malignant" ? "#f87171" : "#4ade80" }}>{record.prediction}</p>
                                                <p style={{ color: "#64748b", fontSize: "0.8rem" }}>{record.confidence.toFixed(1)}%</p>
                                            </div>
                                            {record.reliability && (
                                                <span className={`sharp-badge ${record.reliability === "high" ? "success" : record.reliability === "medium" ? "warning" : "danger"}`} style={{ fontSize: "0.7rem", flexShrink: 0 }}>
                                                    {record.reliability}
                                                </span>
                                            )}
                                            <button onClick={() => handleDelete(record.id)} style={{ padding: "0.5rem", background: "none", border: "none", cursor: "pointer", flexShrink: 0 }}>
                                                <Trash2 style={{ width: "16px", height: "16px", color: "#64748b" }} />
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Pagination */}
                        {totalCount > pageSize && (
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "1rem", marginTop: "1rem" }}>
                                <button onClick={() => setCurrentPage(Math.max(0, currentPage - 1))} disabled={currentPage === 0} style={{ padding: "0.5rem", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", cursor: currentPage === 0 ? "not-allowed" : "pointer", opacity: currentPage === 0 ? 0.5 : 1 }}>
                                    <ChevronLeft style={{ width: "18px", height: "18px", color: "#f1f5f9" }} />
                                </button>
                                <span style={{ color: "#94a3b8", fontSize: "0.9rem" }}>Page {currentPage + 1} of {Math.ceil(totalCount / pageSize)}</span>
                                <button onClick={() => setCurrentPage(currentPage + 1)} disabled={(currentPage + 1) * pageSize >= totalCount} style={{ padding: "0.5rem", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", cursor: (currentPage + 1) * pageSize >= totalCount ? "not-allowed" : "pointer", opacity: (currentPage + 1) * pageSize >= totalCount ? 0.5 : 1 }}>
                                    <ChevronRight style={{ width: "18px", height: "18px", color: "#f1f5f9" }} />
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Batch Upload */}
                    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                        <div className="sharp-card" style={{ padding: "1.5rem" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
                                <Upload style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
                                <h3 style={{ color: "#f1f5f9", fontWeight: 600 }}>Batch Upload</h3>
                            </div>
                            <p style={{ color: "#94a3b8", fontSize: "0.9rem", marginBottom: "1rem" }}>Upload up to 20 images at once for bulk analysis</p>

                            <label style={{ cursor: "pointer", display: "block", marginBottom: "1rem" }}>
                                <div style={{ border: "2px dashed rgba(139, 92, 246, 0.4)", borderRadius: "12px", padding: "1.5rem", textAlign: "center", transition: "all 0.3s" }}>
                                    <Upload style={{ width: "32px", height: "32px", color: "#94a3b8", margin: "0 auto 0.5rem" }} />
                                    <p style={{ color: "#f1f5f9", fontSize: "0.9rem", marginBottom: "0.25rem" }}>Drop files or click</p>
                                    <p style={{ color: "#64748b", fontSize: "0.8rem" }}>Max 20 files</p>
                                </div>
                                <input type="file" multiple accept="image/*" style={{ display: "none" }} onChange={(e) => setBatchFiles(Array.from(e.target.files || []).slice(0, 20))} />
                            </label>

                            {batchFiles.length > 0 && (
                                <div style={{ marginBottom: "1rem" }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                        <span style={{ color: "#f1f5f9", fontSize: "0.9rem" }}>{batchFiles.length} files selected</span>
                                        <button onClick={() => setBatchFiles([])} style={{ color: "#f87171", fontSize: "0.8rem", background: "none", border: "none", cursor: "pointer" }}>Clear</button>
                                    </div>
                                    <div style={{ maxHeight: "120px", overflowY: "auto", background: "rgba(30, 41, 59, 0.5)", borderRadius: "8px", padding: "0.75rem" }}>
                                        {batchFiles.map((file, i) => (
                                            <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8rem", color: "#94a3b8", padding: "0.25rem 0" }}>
                                                <FileImage style={{ width: "12px", height: "12px" }} />
                                                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <button onClick={handleBatchUpload} disabled={batchFiles.length === 0 || batchLoading} className="sharp-btn-primary" style={{ background: "linear-gradient(135deg, #8b5cf6, #7c3aed)" }}>
                                {batchLoading ? <><div className="sharp-spinner" /><span>Analyzing...</span></> : <><Activity style={{ width: "18px", height: "18px" }} /><span>Analyze All</span></>}
                            </button>
                        </div>

                        {/* Batch Result */}
                        {batchResult && (
                            <div className="sharp-card" style={{ padding: "1.5rem", animation: "fadeIn 0.3s ease" }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                                    <h3 style={{ color: "#f1f5f9", fontWeight: 600 }}>Batch Complete</h3>
                                    <button onClick={() => setBatchResult(null)} style={{ background: "none", border: "none", cursor: "pointer" }}>
                                        <X style={{ width: "16px", height: "16px", color: "#94a3b8" }} />
                                    </button>
                                </div>
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
                                    <div style={{ background: "rgba(34, 197, 94, 0.1)", borderRadius: "10px", padding: "1rem", textAlign: "center" }}>
                                        <p style={{ fontSize: "1.5rem", fontWeight: 700, color: "#4ade80" }}>{batchResult.summary.benign_count}</p>
                                        <p style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Benign</p>
                                    </div>
                                    <div style={{ background: "rgba(239, 68, 68, 0.1)", borderRadius: "10px", padding: "1rem", textAlign: "center" }}>
                                        <p style={{ fontSize: "1.5rem", fontWeight: 700, color: "#f87171" }}>{batchResult.summary.malignant_count}</p>
                                        <p style={{ fontSize: "0.8rem", color: "#94a3b8" }}>Malignant</p>
                                    </div>
                                </div>
                                <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>✓ {batchResult.summary.success} analyzed successfully</p>
                                {batchResult.summary.failed > 0 && <p style={{ color: "#f87171", fontSize: "0.85rem" }}>✗ {batchResult.summary.failed} failed</p>}
                            </div>
                        )}

                        {error && (
                            <div className="sharp-card" style={{ background: "rgba(239, 68, 68, 0.1)", borderColor: "rgba(239, 68, 68, 0.3)", padding: "1rem" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                                    <AlertCircle style={{ width: "18px", height: "18px", color: "#f87171" }} />
                                    <p style={{ color: "#fca5a5", fontSize: "0.9rem" }}>{error}</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Mammography - Coming Soon */}
            {activeTab === "mammography" && (
                <>
                    <div className="sharp-info-box" style={{ marginBottom: "2rem", background: "linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.05))", borderColor: "rgba(139, 92, 246, 0.3)" }}>
                        <div className="icon" style={{ background: "rgba(139, 92, 246, 0.2)" }}><Info style={{ width: "20px", height: "20px", color: "#8b5cf6" }} /></div>
                        <div>
                            <h4>Mammography History</h4>
                            <p>Mammography analysis history tracking is coming soon. For now, you can perform mammography predictions on the Mammography Predict page.</p>
                        </div>
                    </div>

                    <div className="sharp-card sharp-placeholder" style={{ padding: "4rem 2rem" }}>
                        <div style={{ width: "100px", height: "100px", background: "rgba(139, 92, 246, 0.15)", borderRadius: "24px", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 2rem" }}>
                            <ScanLine style={{ width: "48px", height: "48px", color: "#8b5cf6" }} />
                        </div>
                        <h2 style={{ color: "#f1f5f9", fontSize: "1.75rem", marginBottom: "0.75rem" }}>Mammography History Coming Soon</h2>
                        <p style={{ color: "#94a3b8", maxWidth: "500px", margin: "0 auto 2rem" }}>
                            We're implementing a dedicated history tracking system for mammography analyses with BI-RADS classification history.
                        </p>

                        <div className="sharp-feature-grid">
                            {[
                                { icon: "📊", title: "BI-RADS Tracking", desc: "Track all BI-RADS classifications over time" },
                                { icon: "📁", title: "Batch Analysis", desc: "Analyze multiple mammograms at once" },
                                { icon: "📈", title: "Statistics", desc: "View analysis statistics and trends" },
                            ].map((feature) => (
                                <div key={feature.title} className="sharp-feature-item">
                                    <div className="icon">{feature.icon}</div>
                                    <h4>{feature.title}</h4>
                                    <p>{feature.desc}</p>
                                </div>
                            ))}
                        </div>

                        <div style={{ marginTop: "2rem" }}>
                            <a href="/mammography" className="sharp-btn-primary" style={{ display: "inline-flex", width: "auto", background: "linear-gradient(135deg, #8b5cf6, #ec4899)" }}>
                                <ScanLine style={{ width: "18px", height: "18px" }} /> Go to Mammography Predict
                            </a>
                        </div>
                    </div>
                </>
            )}

            <style>{`
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
                @keyframes spin { to { transform: rotate(360deg); } }
                @media (max-width: 1024px) {
                    .stats-grid-history { grid-template-columns: repeat(2, 1fr) !important; }
                    .history-grid { grid-template-columns: 1fr !important; }
                }
                @media (max-width: 640px) {
                    .stats-grid-history { grid-template-columns: 1fr !important; }
                }
            `}</style>
        </div>
    );
};

export default History;
