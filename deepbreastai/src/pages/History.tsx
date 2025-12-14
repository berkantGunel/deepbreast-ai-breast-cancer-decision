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

const History = () => {
    const [records, setRecords] = useState<HistoryRecord[]>([]);
    const [stats, setStats] = useState<HistoryStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState("");
    const [predictionFilter, setPredictionFilter] = useState<"Benign" | "Malignant" | "">("");
    const [currentPage, setCurrentPage] = useState(0);
    const [totalCount, setTotalCount] = useState(0);
    const pageSize = 20;

    // Batch upload state
    const [batchFiles, setBatchFiles] = useState<File[]>([]);
    const [batchLoading, setBatchLoading] = useState(false);
    const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);

    // Load history
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

    // Load stats
    const loadStats = async () => {
        try {
            const response = await getHistoryStats();
            setStats(response.statistics);
        } catch (err) {
            console.error("Failed to load stats:", err);
        }
    };

    useEffect(() => {
        loadHistory();
        loadStats();
    }, [currentPage, predictionFilter]);

    // Search with debounce
    useEffect(() => {
        const timeout = setTimeout(() => {
            setCurrentPage(0);
            loadHistory();
        }, 300);
        return () => clearTimeout(timeout);
    }, [searchTerm]);

    // Handle delete
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

    // Handle batch upload
    const handleBatchUpload = async () => {
        if (batchFiles.length === 0) return;
        setBatchLoading(true);
        setBatchResult(null);
        try {
            const result = await batchUpload(batchFiles, {
                useMcDropout: true,
                mcSamples: 20,
            });
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

    // Format date
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
        <div className="page-shell">
            {/* Header */}
            <section className="mb-8">
                <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/30 rounded-2xl">
                        <Clock className="w-8 h-8 text-violet-400" />
                    </div>
                    <div>
                        <h1 className="text-3xl lg:text-4xl font-bold text-white">
                            Analysis History
                        </h1>
                        <p className="text-slate-400 mt-1">
                            View past analyses and upload multiple images
                        </p>
                    </div>
                </div>

                {/* Stats Cards */}
                {stats && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="glass-card p-5 text-center">
                            <BarChart3 className="w-6 h-6 text-violet-400 mx-auto mb-2" />
                            <p className="text-3xl font-bold text-white">{stats.total_analyses}</p>
                            <p className="text-sm text-slate-400">Total Analyses</p>
                        </div>
                        <div className="glass-card p-5 text-center">
                            <CheckCircle className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
                            <p className="text-3xl font-bold text-emerald-400">{stats.by_prediction?.Benign || 0}</p>
                            <p className="text-sm text-slate-400">Benign</p>
                        </div>
                        <div className="glass-card p-5 text-center">
                            <AlertCircle className="w-6 h-6 text-red-400 mx-auto mb-2" />
                            <p className="text-3xl font-bold text-red-400">{stats.by_prediction?.Malignant || 0}</p>
                            <p className="text-sm text-slate-400">Malignant</p>
                        </div>
                        <div className="glass-card p-5 text-center">
                            <TrendingUp className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
                            <p className="text-3xl font-bold text-cyan-400">{stats.average_confidence}%</p>
                            <p className="text-sm text-slate-400">Avg Confidence</p>
                        </div>
                    </div>
                )}
            </section>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* History List - 2 columns */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Search & Filter */}
                    <div className="glass-card p-4">
                        <div className="flex flex-col sm:flex-row gap-4">
                            <div className="flex-1 relative">
                                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
                                <input
                                    type="text"
                                    placeholder="Search files..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="w-full pl-10 pr-4 py-2.5 bg-slate-800/50 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-violet-500"
                                />
                            </div>
                            <div className="flex items-center gap-2">
                                <Filter className="w-5 h-5 text-slate-400 hidden sm:block" />
                                <select
                                    value={predictionFilter}
                                    onChange={(e) => setPredictionFilter(e.target.value as "" | "Benign" | "Malignant")}
                                    className="px-4 py-2.5 bg-slate-800/50 border border-white/10 rounded-xl text-white focus:outline-none focus:border-violet-500"
                                    style={{ colorScheme: 'dark' }}
                                >
                                    <option value="">All Results</option>
                                    <option value="Benign">Benign Only</option>
                                    <option value="Malignant">Malignant Only</option>
                                </select>
                                <button
                                    onClick={() => { loadHistory(); loadStats(); }}
                                    className="p-2.5 bg-slate-800/50 border border-white/10 rounded-xl hover:bg-white/10 transition-colors"
                                    title="Refresh"
                                >
                                    <RefreshCw className={`w-5 h-5 text-slate-400 ${loading ? 'animate-spin' : ''}`} />
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Records List */}
                    {loading && records.length === 0 ? (
                        <div className="glass-card p-12 text-center">
                            <Loader2 className="w-10 h-10 text-violet-400 animate-spin mx-auto mb-4" />
                            <p className="text-slate-400">Loading history...</p>
                        </div>
                    ) : records.length === 0 ? (
                        <div className="glass-card p-12 text-center">
                            <Clock className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                            <h3 className="text-lg font-semibold text-slate-300 mb-2">No Records Found</h3>
                            <p className="text-slate-500">Analyze some images to see them here</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {records.map((record) => (
                                <div
                                    key={record.id}
                                    className={`glass-card p-4 hover:border-violet-500/30 transition-colors ${record.prediction === "Malignant"
                                            ? "border-l-4 border-l-red-500"
                                            : "border-l-4 border-l-emerald-500"
                                        }`}
                                >
                                    <div className="flex items-center gap-4">
                                        {/* Thumbnail */}
                                        {record.thumbnail ? (
                                            <img
                                                src={`data:image/jpeg;base64,${record.thumbnail}`}
                                                alt={record.filename}
                                                className="w-14 h-14 rounded-lg object-cover flex-shrink-0"
                                            />
                                        ) : (
                                            <div className="w-14 h-14 bg-white/5 rounded-lg flex items-center justify-center flex-shrink-0">
                                                <FileImage className="w-6 h-6 text-slate-500" />
                                            </div>
                                        )}

                                        {/* Info */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="font-medium text-white truncate text-sm">{record.filename}</h4>
                                                {record.is_batch && (
                                                    <span className="px-2 py-0.5 bg-violet-500/20 text-violet-400 text-xs rounded-full">Batch</span>
                                                )}
                                            </div>
                                            <p className="text-xs text-slate-400">{formatDate(record.created_at)}</p>
                                        </div>

                                        {/* Result */}
                                        <div className="text-right flex-shrink-0">
                                            <p className={`font-semibold text-sm ${record.prediction === "Malignant" ? "text-red-400" : "text-emerald-400"
                                                }`}>
                                                {record.prediction}
                                            </p>
                                            <p className="text-xs text-slate-400">{record.confidence.toFixed(1)}%</p>
                                        </div>

                                        {/* Reliability Badge */}
                                        {record.reliability && (
                                            <div className={`px-2 py-1 rounded-full text-xs font-medium flex-shrink-0 hidden sm:block ${record.reliability === "high"
                                                    ? "bg-emerald-500/20 text-emerald-400"
                                                    : record.reliability === "medium"
                                                        ? "bg-amber-500/20 text-amber-400"
                                                        : "bg-red-500/20 text-red-400"
                                                }`}>
                                                {record.reliability}
                                            </div>
                                        )}

                                        {/* Delete Button */}
                                        <button
                                            onClick={() => handleDelete(record.id)}
                                            className="p-2 hover:bg-red-500/10 rounded-lg transition-colors group flex-shrink-0"
                                        >
                                            <Trash2 className="w-4 h-4 text-slate-500 group-hover:text-red-400" />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Pagination */}
                    {totalCount > pageSize && (
                        <div className="flex items-center justify-center gap-4 mt-6">
                            <button
                                onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                                disabled={currentPage === 0}
                                className="p-2 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                            >
                                <ChevronLeft className="w-5 h-5" />
                            </button>
                            <span className="text-slate-400 text-sm">
                                Page {currentPage + 1} of {Math.ceil(totalCount / pageSize)}
                            </span>
                            <button
                                onClick={() => setCurrentPage(currentPage + 1)}
                                disabled={(currentPage + 1) * pageSize >= totalCount}
                                className="p-2 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed text-white"
                            >
                                <ChevronRight className="w-5 h-5" />
                            </button>
                        </div>
                    )}
                </div>

                {/* Batch Upload - 1 column */}
                <div className="space-y-6">
                    <div className="glass-card p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <Upload className="w-5 h-5 text-violet-400" />
                            <h3 className="text-lg font-semibold text-white">Batch Upload</h3>
                        </div>

                        <p className="text-sm text-slate-400 mb-4">
                            Upload up to 20 images at once for bulk analysis
                        </p>

                        {/* File Input */}
                        <label className="block cursor-pointer mb-4">
                            <div className="border-2 border-dashed border-white/10 rounded-xl p-6 text-center hover:border-violet-500/30 transition-colors">
                                <Upload className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                                <p className="text-sm text-white mb-1">Drop files or click to browse</p>
                                <p className="text-xs text-slate-500">Max 20 files</p>
                            </div>
                            <input
                                type="file"
                                multiple
                                accept="image/*"
                                className="hidden"
                                onChange={(e) => {
                                    const files = Array.from(e.target.files || []);
                                    setBatchFiles(files.slice(0, 20));
                                }}
                            />
                        </label>

                        {/* Selected Files */}
                        {batchFiles.length > 0 && (
                            <div className="mb-4">
                                <div className="flex items-center justify-between mb-2">
                                    <p className="text-sm text-white">{batchFiles.length} files selected</p>
                                    <button
                                        onClick={() => setBatchFiles([])}
                                        className="text-xs text-red-400 hover:text-red-300"
                                    >
                                        Clear
                                    </button>
                                </div>
                                <div className="max-h-32 overflow-y-auto space-y-1 bg-slate-800/30 rounded-lg p-2">
                                    {batchFiles.map((file, i) => (
                                        <div key={i} className="flex items-center gap-2 text-xs text-slate-400">
                                            <FileImage className="w-3 h-3 flex-shrink-0" />
                                            <span className="truncate">{file.name}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Upload Button */}
                        <button
                            onClick={handleBatchUpload}
                            disabled={batchFiles.length === 0 || batchLoading}
                            className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-gradient-to-r from-violet-500 to-purple-500 text-white font-medium rounded-xl hover:from-violet-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            {batchLoading ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>Analyzing...</span>
                                </>
                            ) : (
                                <>
                                    <Activity className="w-4 h-4" />
                                    <span>Analyze All</span>
                                </>
                            )}
                        </button>
                    </div>

                    {/* Batch Result */}
                    {batchResult && (
                        <div className="glass-card p-6 animate-fade-in">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="font-semibold text-white">Batch Complete</h3>
                                <button
                                    onClick={() => setBatchResult(null)}
                                    className="p-1 hover:bg-white/5 rounded"
                                >
                                    <X className="w-4 h-4 text-slate-400" />
                                </button>
                            </div>

                            <div className="grid grid-cols-2 gap-3 mb-4">
                                <div className="bg-emerald-500/10 rounded-xl p-3 text-center">
                                    <p className="text-2xl font-bold text-emerald-400">{batchResult.summary.benign_count}</p>
                                    <p className="text-xs text-slate-400">Benign</p>
                                </div>
                                <div className="bg-red-500/10 rounded-xl p-3 text-center">
                                    <p className="text-2xl font-bold text-red-400">{batchResult.summary.malignant_count}</p>
                                    <p className="text-xs text-slate-400">Malignant</p>
                                </div>
                            </div>

                            <div className="text-sm text-slate-400">
                                <p>✓ {batchResult.summary.success} analyzed successfully</p>
                                {batchResult.summary.failed > 0 && (
                                    <p className="text-red-400">✗ {batchResult.summary.failed} failed</p>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Error Display */}
                    {error && (
                        <div className="glass-card bg-red-500/10 border-red-500/30 p-4">
                            <div className="flex items-center gap-3">
                                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                                <p className="text-red-300 text-sm">{error}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default History;
