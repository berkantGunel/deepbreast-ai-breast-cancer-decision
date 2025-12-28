import { useState, useEffect } from "react";
import {
    BarChart3,
    Activity,
    TrendingUp,
    Users,
    CheckCircle,
    AlertCircle,
    Clock,
    Brain,
    ScanLine,
    RefreshCw,
    Download,
    FileSpreadsheet,
    FileText,
    ChevronDown
} from "lucide-react";
import { useToast } from "../components/Toast";
import { exportToCSV, exportToExcel, exportSummaryReport } from "../utils/exportData";
import { getHistoryStats, getHistory } from "../services/api";

interface AnalysisRecord {
    id: string;
    type: "histopathology" | "mammography";
    prediction: string;
    confidence: number;
    timestamp: string;
}

interface DashboardStats {
    totalAnalyses: number;
    histopathologyCount: number;
    mammographyCount: number;
    benignCount: number;
    malignantCount: number;
    suspiciousCount: number;
    avgConfidence: number;
    todayCount: number;
    weekCount: number;
}

const Dashboard = () => {
    const { showToast } = useToast();
    const [stats, setStats] = useState<DashboardStats>({
        totalAnalyses: 0,
        histopathologyCount: 0,
        mammographyCount: 0,
        benignCount: 0,
        malignantCount: 0,
        suspiciousCount: 0,
        avgConfidence: 0,
        todayCount: 0,
        weekCount: 0,
    });
    const [recentAnalyses, setRecentAnalyses] = useState<AnalysisRecord[]>([]);
    const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking");
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [exportDropdownOpen, setExportDropdownOpen] = useState(false);

    // Export handlers - now using backend API
    const handleExportCSV = async () => {
        try {
            // Fetch all history from backend (limit high to get all)
            const response = await getHistory({ limit: 200 });
            const data: AnalysisRecord[] = response.records.map(record => ({
                id: record.id.toString(),
                type: "histopathology" as const,
                prediction: record.prediction,
                confidence: record.confidence,
                timestamp: record.created_at
            }));

            if (data.length === 0) {
                showToast('No analysis data to export', 'warning');
                return;
            }
            exportToCSV(data, 'deepbreast_analysis');
            showToast('CSV file downloaded successfully!', 'success');
            setExportDropdownOpen(false);
        } catch (error) {
            console.error('CSV export error:', error);
            showToast('Failed to export CSV file', 'error');
        }
    };

    const handleExportExcel = async () => {
        try {
            // Fetch all history from backend
            const response = await getHistory({ limit: 200 });
            const data: AnalysisRecord[] = response.records.map(record => ({
                id: record.id.toString(),
                type: "histopathology" as const,
                prediction: record.prediction,
                confidence: record.confidence,
                timestamp: record.created_at
            }));

            if (data.length === 0) {
                showToast('No analysis data to export', 'warning');
                return;
            }
            exportToExcel(data, 'deepbreast_analysis');
            showToast('Excel file downloaded successfully!', 'success');
            setExportDropdownOpen(false);
        } catch (error) {
            console.error('Excel export error:', error);
            showToast('Failed to export Excel file', 'error');
        }
    };

    const handleExportSummary = async () => {
        try {
            // Fetch all history from backend
            const response = await getHistory({ limit: 200 });
            const data: AnalysisRecord[] = response.records.map(record => ({
                id: record.id.toString(),
                type: "histopathology" as const,
                prediction: record.prediction,
                confidence: record.confidence,
                timestamp: record.created_at
            }));

            if (data.length === 0) {
                showToast('No analysis data to export', 'warning');
                return;
            }
            exportSummaryReport(data);
            showToast('Summary report downloaded!', 'success');
            setExportDropdownOpen(false);
        } catch (error) {
            console.error('Summary export error:', error);
            showToast('Failed to export summary report', 'error');
        }
    };

    // Load stats from backend API
    const loadStats = async () => {
        try {
            // Load statistics from backend
            const statsResponse = await getHistoryStats();
            const apiStats = statsResponse.statistics;

            // Load recent history for the list
            const historyResponse = await getHistory({ limit: 10 });

            // Convert API records to dashboard format
            const recentRecords: AnalysisRecord[] = historyResponse.records.map(record => ({
                id: record.id.toString(),
                type: "histopathology" as const, // Histopathology records from API
                prediction: record.prediction,
                confidence: record.confidence,
                timestamp: record.created_at
            }));

            setRecentAnalyses(recentRecords);

            // Calculate time-based counts from daily_counts
            const now = new Date();
            const today = now.toISOString().split('T')[0];
            const todayCount = apiStats.daily_counts?.[today] || 0;

            // Sum last 7 days
            let weekCount = 0;
            for (let i = 0; i < 7; i++) {
                const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
                const dateStr = date.toISOString().split('T')[0];
                weekCount += apiStats.daily_counts?.[dateStr] || 0;
            }

            setStats({
                totalAnalyses: apiStats.total_analyses,
                histopathologyCount: apiStats.total_analyses, // All are histopathology from this API
                mammographyCount: 0, // Mammography history not yet implemented
                benignCount: apiStats.by_prediction?.Benign || 0,
                malignantCount: apiStats.by_prediction?.Malignant || 0,
                suspiciousCount: apiStats.by_prediction?.Suspicious || 0,
                avgConfidence: apiStats.average_confidence || 0,
                todayCount,
                weekCount,
            });
        } catch (error) {
            console.error("Error loading stats from API:", error);
            // Fallback: keep current stats or show zeros
        }
    };

    // Check API status
    const checkApiStatus = async () => {
        try {
            const response = await fetch("http://localhost:8000/api/health");
            if (response.ok) {
                setApiStatus("online");
            } else {
                setApiStatus("offline");
            }
        } catch {
            setApiStatus("offline");
        }
    };

    const handleRefresh = async () => {
        setIsRefreshing(true);
        loadStats();
        await checkApiStatus();
        setTimeout(() => setIsRefreshing(false), 500);
    };

    useEffect(() => {
        loadStats();
        checkApiStatus();
        // Refresh every 30 seconds
        const interval = setInterval(() => {
            checkApiStatus();
        }, 30000);
        return () => clearInterval(interval);
    }, []);

    const getPredictionColor = (prediction: string) => {
        const pred = prediction.toLowerCase();
        if (pred.includes("benign")) return "#22c55e";
        if (pred.includes("malignant")) return "#ef4444";
        if (pred.includes("suspicious")) return "#f59e0b";
        return "#6b7280";
    };

    const formatTimeAgo = (timestamp: string) => {
        const diff = Date.now() - new Date(timestamp).getTime();
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return "Just now";
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        return `${days}d ago`;
    };

    return (
        <div className="sharp-page">
            {/* Header */}
            <section className="sharp-section">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
                    <div className="sharp-header" style={{ marginBottom: 0 }}>
                        <h1 style={{ fontSize: "2rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
                            <BarChart3 style={{ width: "32px", height: "32px", color: "#8b5cf6" }} />
                            Dashboard
                        </h1>
                        <p className="subtitle">System overview and analysis statistics</p>
                    </div>
                    <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
                        {/* Export Dropdown */}
                        <div style={{ position: "relative" }}>
                            <button
                                onClick={() => setExportDropdownOpen(!exportDropdownOpen)}
                                className="sharp-btn-secondary"
                                style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                            >
                                <Download style={{ width: "18px", height: "18px" }} />
                                Export
                                <ChevronDown style={{
                                    width: "16px",
                                    height: "16px",
                                    transition: "transform 0.2s",
                                    transform: exportDropdownOpen ? "rotate(180deg)" : "rotate(0deg)"
                                }} />
                            </button>
                            {exportDropdownOpen && (
                                <>
                                    {/* Backdrop to close dropdown */}
                                    <div
                                        style={{
                                            position: "fixed",
                                            top: 0,
                                            left: 0,
                                            right: 0,
                                            bottom: 0,
                                            zIndex: 49
                                        }}
                                        onClick={() => setExportDropdownOpen(false)}
                                    />
                                    <div style={{
                                        position: "absolute",
                                        top: "calc(100% + 8px)",
                                        right: 0,
                                        background: "rgba(30, 41, 59, 0.98)",
                                        backdropFilter: "blur(12px)",
                                        border: "1px solid rgba(255, 255, 255, 0.1)",
                                        borderRadius: "12px",
                                        padding: "0.5rem",
                                        minWidth: "200px",
                                        zIndex: 50,
                                        boxShadow: "0 10px 40px rgba(0, 0, 0, 0.4)",
                                        animation: "fadeIn 0.2s ease"
                                    }}>
                                        <button
                                            onClick={handleExportCSV}
                                            style={{
                                                display: "flex",
                                                alignItems: "center",
                                                gap: "0.75rem",
                                                width: "100%",
                                                padding: "0.75rem 1rem",
                                                background: "transparent",
                                                border: "none",
                                                borderRadius: "8px",
                                                color: "#f1f5f9",
                                                fontSize: "0.9rem",
                                                cursor: "pointer",
                                                transition: "background 0.2s"
                                            }}
                                            onMouseOver={(e) => e.currentTarget.style.background = "rgba(139, 92, 246, 0.15)"}
                                            onMouseOut={(e) => e.currentTarget.style.background = "transparent"}
                                        >
                                            <FileText style={{ width: "18px", height: "18px", color: "#22c55e" }} />
                                            Export as CSV
                                        </button>
                                        <button
                                            onClick={handleExportExcel}
                                            style={{
                                                display: "flex",
                                                alignItems: "center",
                                                gap: "0.75rem",
                                                width: "100%",
                                                padding: "0.75rem 1rem",
                                                background: "transparent",
                                                border: "none",
                                                borderRadius: "8px",
                                                color: "#f1f5f9",
                                                fontSize: "0.9rem",
                                                cursor: "pointer",
                                                transition: "background 0.2s"
                                            }}
                                            onMouseOver={(e) => e.currentTarget.style.background = "rgba(139, 92, 246, 0.15)"}
                                            onMouseOut={(e) => e.currentTarget.style.background = "transparent"}
                                        >
                                            <FileSpreadsheet style={{ width: "18px", height: "18px", color: "#06b6d4" }} />
                                            Export as Excel
                                        </button>
                                        <div style={{ height: "1px", background: "rgba(255,255,255,0.1)", margin: "0.5rem 0" }} />
                                        <button
                                            onClick={handleExportSummary}
                                            style={{
                                                display: "flex",
                                                alignItems: "center",
                                                gap: "0.75rem",
                                                width: "100%",
                                                padding: "0.75rem 1rem",
                                                background: "transparent",
                                                border: "none",
                                                borderRadius: "8px",
                                                color: "#f1f5f9",
                                                fontSize: "0.9rem",
                                                cursor: "pointer",
                                                transition: "background 0.2s"
                                            }}
                                            onMouseOver={(e) => e.currentTarget.style.background = "rgba(139, 92, 246, 0.15)"}
                                            onMouseOut={(e) => e.currentTarget.style.background = "transparent"}
                                        >
                                            <BarChart3 style={{ width: "18px", height: "18px", color: "#f59e0b" }} />
                                            Summary Report
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>
                        <button
                            onClick={handleRefresh}
                            className="sharp-btn-secondary"
                            style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                        >
                            <RefreshCw style={{ width: "18px", height: "18px", animation: isRefreshing ? "spin 1s linear infinite" : "none" }} />
                            Refresh
                        </button>
                    </div>
                </div>

                {/* System Status */}
                <div className="sharp-card" style={{
                    padding: "1rem 1.5rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "1rem",
                    marginBottom: "1.5rem",
                    borderLeft: `3px solid ${apiStatus === "online" ? "#22c55e" : apiStatus === "offline" ? "#ef4444" : "#f59e0b"}`
                }}>
                    <div style={{
                        width: "12px",
                        height: "12px",
                        borderRadius: "50%",
                        background: apiStatus === "online" ? "#22c55e" : apiStatus === "offline" ? "#ef4444" : "#f59e0b",
                        animation: apiStatus === "checking" ? "pulse 1s infinite" : "none"
                    }} />
                    <div>
                        <span style={{ color: "#f1f5f9", fontWeight: 600 }}>API Status: </span>
                        <span style={{
                            color: apiStatus === "online" ? "#4ade80" : apiStatus === "offline" ? "#f87171" : "#fbbf24",
                            fontWeight: 600
                        }}>
                            {apiStatus === "online" ? "Online" : apiStatus === "offline" ? "Offline" : "Checking..."}
                        </span>
                    </div>
                    {apiStatus === "online" && (
                        <span style={{ marginLeft: "auto", color: "#94a3b8", fontSize: "0.875rem" }}>
                            All systems operational
                        </span>
                    )}
                </div>
            </section>

            {/* Stats Grid */}
            <section className="sharp-section">
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem" }} className="dashboard-stats-grid">
                    {/* Total Analyses */}
                    <div className="sharp-metric-card">
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                            <Activity style={{ width: "24px", height: "24px", color: "#8b5cf6" }} />
                            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>All time</span>
                        </div>
                        <div className="sharp-metric-value" style={{ color: "#8b5cf6" }}>{stats.totalAnalyses}</div>
                        <div className="sharp-metric-label">Total Analyses</div>
                    </div>

                    {/* Today */}
                    <div className="sharp-metric-card">
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                            <Clock style={{ width: "24px", height: "24px", color: "#06b6d4" }} />
                            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Today</span>
                        </div>
                        <div className="sharp-metric-value" style={{ color: "#06b6d4" }}>{stats.todayCount}</div>
                        <div className="sharp-metric-label">Today's Analyses</div>
                    </div>

                    {/* This Week */}
                    <div className="sharp-metric-card">
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                            <TrendingUp style={{ width: "24px", height: "24px", color: "#10b981" }} />
                            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>7 days</span>
                        </div>
                        <div className="sharp-metric-value" style={{ color: "#10b981" }}>{stats.weekCount}</div>
                        <div className="sharp-metric-label">This Week</div>
                    </div>

                    {/* Avg Confidence */}
                    <div className="sharp-metric-card">
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                            <CheckCircle style={{ width: "24px", height: "24px", color: "#f59e0b" }} />
                            <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Average</span>
                        </div>
                        <div className="sharp-metric-value" style={{ color: "#f59e0b" }}>
                            {stats.avgConfidence.toFixed(1)}%
                        </div>
                        <div className="sharp-metric-label">Avg Confidence</div>
                    </div>
                </div>
            </section>

            {/* Analysis Breakdown */}
            <section className="sharp-section">
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }} className="dashboard-breakdown-grid">
                    {/* By Model Type */}
                    <div className="sharp-card" style={{ padding: "1.5rem" }}>
                        <h3 style={{ color: "#f1f5f9", fontWeight: 600, marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                            <Users style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
                            Analysis by Model
                        </h3>
                        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                            {/* Histopathology */}
                            <div>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                    <span style={{ color: "#94a3b8", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                                        <Brain style={{ width: "16px", height: "16px", color: "#10b981" }} />
                                        Histopathology
                                    </span>
                                    <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{stats.histopathologyCount}</span>
                                </div>
                                <div style={{ height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden" }}>
                                    <div style={{
                                        height: "100%",
                                        width: `${stats.totalAnalyses > 0 ? (stats.histopathologyCount / stats.totalAnalyses) * 100 : 0}%`,
                                        background: "linear-gradient(90deg, #10b981, #06b6d4)",
                                        borderRadius: "4px",
                                        transition: "width 0.5s ease"
                                    }} />
                                </div>
                            </div>

                            {/* Mammography */}
                            <div>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                    <span style={{ color: "#94a3b8", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                                        <ScanLine style={{ width: "16px", height: "16px", color: "#8b5cf6" }} />
                                        Mammography
                                    </span>
                                    <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{stats.mammographyCount}</span>
                                </div>
                                <div style={{ height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden" }}>
                                    <div style={{
                                        height: "100%",
                                        width: `${stats.totalAnalyses > 0 ? (stats.mammographyCount / stats.totalAnalyses) * 100 : 0}%`,
                                        background: "linear-gradient(90deg, #8b5cf6, #ec4899)",
                                        borderRadius: "4px",
                                        transition: "width 0.5s ease"
                                    }} />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* By Prediction */}
                    <div className="sharp-card" style={{ padding: "1.5rem" }}>
                        <h3 style={{ color: "#f1f5f9", fontWeight: 600, marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                            <AlertCircle style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
                            Analysis by Result
                        </h3>
                        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                            {/* Benign */}
                            <div>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                    <span style={{ color: "#4ade80" }}>Benign</span>
                                    <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{stats.benignCount}</span>
                                </div>
                                <div style={{ height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden" }}>
                                    <div style={{
                                        height: "100%",
                                        width: `${stats.totalAnalyses > 0 ? (stats.benignCount / stats.totalAnalyses) * 100 : 0}%`,
                                        background: "#22c55e",
                                        borderRadius: "4px",
                                        transition: "width 0.5s ease"
                                    }} />
                                </div>
                            </div>

                            {/* Suspicious */}
                            <div>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                    <span style={{ color: "#fbbf24" }}>Suspicious</span>
                                    <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{stats.suspiciousCount}</span>
                                </div>
                                <div style={{ height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden" }}>
                                    <div style={{
                                        height: "100%",
                                        width: `${stats.totalAnalyses > 0 ? (stats.suspiciousCount / stats.totalAnalyses) * 100 : 0}%`,
                                        background: "#f59e0b",
                                        borderRadius: "4px",
                                        transition: "width 0.5s ease"
                                    }} />
                                </div>
                            </div>

                            {/* Malignant */}
                            <div>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                                    <span style={{ color: "#f87171" }}>Malignant</span>
                                    <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{stats.malignantCount}</span>
                                </div>
                                <div style={{ height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden" }}>
                                    <div style={{
                                        height: "100%",
                                        width: `${stats.totalAnalyses > 0 ? (stats.malignantCount / stats.totalAnalyses) * 100 : 0}%`,
                                        background: "#ef4444",
                                        borderRadius: "4px",
                                        transition: "width 0.5s ease"
                                    }} />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Recent Analyses */}
            <section className="sharp-section">
                <div className="sharp-card" style={{ padding: "1.5rem" }}>
                    <h3 style={{ color: "#f1f5f9", fontWeight: 600, marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <Clock style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
                        Recent Analyses
                    </h3>

                    {recentAnalyses.length === 0 ? (
                        <div style={{ textAlign: "center", padding: "2rem", color: "#94a3b8" }}>
                            <Activity style={{ width: "48px", height: "48px", margin: "0 auto 1rem", opacity: 0.5 }} />
                            <p>No analyses yet. Start by uploading an image!</p>
                        </div>
                    ) : (
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                            {recentAnalyses.map((record, index) => (
                                <div
                                    key={record.id || index}
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "1rem",
                                        padding: "1rem",
                                        background: "rgba(255,255,255,0.03)",
                                        borderRadius: "12px",
                                        border: "1px solid rgba(255,255,255,0.05)"
                                    }}
                                >
                                    <div style={{
                                        width: "40px",
                                        height: "40px",
                                        borderRadius: "10px",
                                        background: record.type === "histopathology" ? "rgba(16, 185, 129, 0.15)" : "rgba(139, 92, 246, 0.15)",
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center"
                                    }}>
                                        {record.type === "histopathology" ? (
                                            <Brain style={{ width: "20px", height: "20px", color: "#10b981" }} />
                                        ) : (
                                            <ScanLine style={{ width: "20px", height: "20px", color: "#8b5cf6" }} />
                                        )}
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                                            <span style={{ color: "#f1f5f9", fontWeight: 600 }}>{record.prediction}</span>
                                            <span style={{
                                                fontSize: "0.75rem",
                                                padding: "0.2rem 0.5rem",
                                                background: `${getPredictionColor(record.prediction)}20`,
                                                color: getPredictionColor(record.prediction),
                                                borderRadius: "4px"
                                            }}>
                                                {record.confidence.toFixed(1)}%
                                            </span>
                                        </div>
                                        <div style={{ fontSize: "0.875rem", color: "#94a3b8" }}>
                                            {record.type === "histopathology" ? "Histopathology" : "Mammography"}
                                        </div>
                                    </div>
                                    <div style={{ fontSize: "0.875rem", color: "#64748b" }}>
                                        {formatTimeAgo(record.timestamp)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </section>

            <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (max-width: 1024px) {
          .dashboard-stats-grid { grid-template-columns: repeat(2, 1fr) !important; }
          .dashboard-breakdown-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 640px) {
          .dashboard-stats-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
        </div>
    );
};

export default Dashboard;
