import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
    Upload,
    Scan,
    Target,
    Activity,
    AlertCircle,
    CheckCircle,
    Loader2,
    Download,
    Layers,
    Circle,
    Square,
    ArrowDownToLine,
    RefreshCw,
    Info,
    SlidersHorizontal,
    GripVertical,
    Eye,
    Image as ImageIcon,
    Ruler,
    FileText,
    Crosshair,
    Move,
    Trash2
} from 'lucide-react';
import ImageViewer from '../components/ImageViewer';
import './Segmentation.css';

interface DicomMetadata {
    patient?: {
        id: string;
        name: string;
        birth_date: string;
        sex: string;
        age: string;
    };
    study?: {
        id: string;
        date: string;
        description: string;
    };
    series?: {
        modality: string;
        body_part: string;
        description: string;
    };
    image?: {
        rows: number;
        columns: number;
        pixel_spacing: number[];
    };
    mammography?: {
        laterality: string;
        view_position: string;
    };
}

interface SegmentationResult {
    success: boolean;
    inference_time_ms: number;
    image_size: { width: number; height: number };
    threshold: number;
    pixel_spacing?: number;
    metrics: {
        tumor_detected: boolean;
        num_lesions: number;
        total_area_mm2: number;
        largest_area_mm2: number;
        total_perimeter_mm: number;
        centroids?: Array<{ x: number; y: number }>;
        bounding_boxes?: Array<{ x: number; y: number; width: number; height: number }>;
    };
    mask: string;
    overlay: string;
    original: string;  // Original image for compare mode
    contours?: Array<Array<[number, number]>>;
    dicom_metadata?: DicomMetadata | null;
}

const Segmentation: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [result, setResult] = useState<SegmentationResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [threshold, setThreshold] = useState(0.3);
    const [viewMode, setViewMode] = useState<'overlay' | 'mask' | 'original' | 'compare'>('overlay');
    const [pixelSpacing, setPixelSpacing] = useState(1.0);
    const [overlayOpacity, setOverlayOpacity] = useState(0.5);
    const [comparePosition, setComparePosition] = useState(50);
    const [imageViewMode, setImageViewMode] = useState<'simple' | 'zoom'>('simple');

    // New features state
    const [showContours, setShowContours] = useState(true);
    const [contourAnimated, setContourAnimated] = useState(true);
    const [measurementMode, setMeasurementMode] = useState<'none' | 'distance' | 'area'>('none');
    const [measurements, setMeasurements] = useState<Array<{
        type: 'distance' | 'area';
        points: Array<{ x: number; y: number }>;
        value: number;
    }>>([]);
    const [currentMeasurement, setCurrentMeasurement] = useState<Array<{ x: number; y: number }>>([]);
    const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const compareRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imageContainerRef = useRef<HTMLDivElement>(null);

    const getApiUrl = () => {
        if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
            return `http://${window.location.hostname}:8000`;
        }
        return 'http://localhost:8000';
    };

    // Check if file is TIFF format
    const isTiffFile = (file: File) => {
        return file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');
    };

    // Check if file is DICOM format
    const isDicomFile = (file: File) => {
        return file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom');
    };

    // Check if file needs special handling (no browser preview)
    const isSpecialFile = (file: File) => isTiffFile(file) || isDicomFile(file);

    const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setResult(null);
            setError(null);

            // For TIFF/DICOM files, we can't show preview in browser
            // Set previewUrl to null, we'll show a placeholder instead
            if (isSpecialFile(file)) {
                setPreviewUrl(null);
            } else {
                setPreviewUrl(URL.createObjectURL(file));
            }
        }
    }, []);

    const handleDrop = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        // Accept image files and special formats (which may not have image/ MIME type)
        if (file && (file.type.startsWith('image/') || isSpecialFile(file))) {
            setSelectedFile(file);
            setResult(null);
            setError(null);

            if (isSpecialFile(file)) {
                setPreviewUrl(null);
            } else {
                setPreviewUrl(URL.createObjectURL(file));
            }
        }
    }, []);

    const handleDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
    }, []);

    const handleSegment = async () => {
        if (!selectedFile) return;

        setIsLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(
                `${getApiUrl()}/api/segmentation/predict?threshold=${threshold}&return_overlay=true&return_contours=true&pixel_spacing=${pixelSpacing}&overlay_alpha=${overlayOpacity}`,
                {
                    method: 'POST',
                    body: formData,
                }
            );

            if (!response.ok) {
                throw new Error(`Segmentation failed: ${response.statusText}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Segmentation failed');
        } finally {
            setIsLoading(false);
        }
    };

    const handleDownloadMask = () => {
        if (!result?.mask) return;

        const link = document.createElement('a');
        link.href = `data:image/png;base64,${result.mask}`;
        link.download = 'tumor_mask.png';
        link.click();
    };

    const handleDownloadOverlay = () => {
        if (!result?.overlay) return;

        const link = document.createElement('a');
        link.href = `data:image/png;base64,${result.overlay}`;
        link.download = 'tumor_overlay.png';
        link.click();
    };

    const handleReset = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResult(null);
        setError(null);
        setMeasurements([]);
        setCurrentMeasurement([]);
        setMeasurementMode('none');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Calculate distance between two points
    const calculateDistance = (p1: { x: number; y: number }, p2: { x: number; y: number }) => {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        return Math.sqrt(dx * dx + dy * dy) * pixelSpacing;
    };

    // Calculate area of polygon
    const calculatePolygonArea = (points: Array<{ x: number; y: number }>) => {
        if (points.length < 3) return 0;
        let area = 0;
        for (let i = 0; i < points.length; i++) {
            const j = (i + 1) % points.length;
            area += points[i].x * points[j].y;
            area -= points[j].x * points[i].y;
        }
        return Math.abs(area / 2) * pixelSpacing * pixelSpacing;
    };

    // Handle measurement click
    const handleMeasurementClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (measurementMode === 'none' || !imageContainerRef.current) return;

        const rect = imageContainerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const newPoint = { x, y };
        const newMeasurement = [...currentMeasurement, newPoint];
        setCurrentMeasurement(newMeasurement);

        if (measurementMode === 'distance' && newMeasurement.length === 2) {
            const distance = calculateDistance(newMeasurement[0], newMeasurement[1]);
            setMeasurements([...measurements, {
                type: 'distance',
                points: newMeasurement,
                value: distance
            }]);
            setCurrentMeasurement([]);
        } else if (measurementMode === 'area' && newMeasurement.length >= 3) {
            // Double-click to complete area measurement
            if (e.detail === 2) {
                const area = calculatePolygonArea(newMeasurement);
                setMeasurements([...measurements, {
                    type: 'area',
                    points: newMeasurement,
                    value: area
                }]);
                setCurrentMeasurement([]);
            }
        }
    };

    // Clear all measurements
    const clearMeasurements = () => {
        setMeasurements([]);
        setCurrentMeasurement([]);
    };

    // Generate PDF Report
    const handleGeneratePDF = async () => {
        if (!result) return;

        setIsGeneratingPDF(true);

        try {
            // Dynamically import jspdf
            const { jsPDF } = await import('jspdf');
            const doc = new jsPDF();

            const pageWidth = doc.internal.pageSize.getWidth();
            let yPos = 20;

            // Header
            doc.setFillColor(16, 185, 129);
            doc.rect(0, 0, pageWidth, 40, 'F');

            doc.setTextColor(255, 255, 255);
            doc.setFontSize(24);
            doc.setFont('helvetica', 'bold');
            doc.text('DeepBreast AI', 20, 25);

            doc.setFontSize(12);
            doc.setFont('helvetica', 'normal');
            doc.text('Tumor Segmentation Analysis Report', 20, 35);

            yPos = 55;

            // Report Info
            doc.setTextColor(0, 0, 0);
            doc.setFontSize(10);
            doc.setFont('helvetica', 'normal');

            const reportDate = new Date().toLocaleString();
            doc.text(`Report Generated: ${reportDate}`, 20, yPos);
            yPos += 8;
            doc.text(`File: ${selectedFile?.name || 'Unknown'}`, 20, yPos);
            yPos += 15;

            // Detection Status
            doc.setFontSize(16);
            doc.setFont('helvetica', 'bold');

            if (result.metrics.tumor_detected) {
                doc.setTextColor(220, 38, 38);
                doc.text('⚠ TUMOR DETECTED', 20, yPos);
            } else {
                doc.setTextColor(34, 197, 94);
                doc.text('✓ NO TUMOR DETECTED', 20, yPos);
            }
            yPos += 15;

            // Metrics Section
            doc.setTextColor(0, 0, 0);
            doc.setFontSize(14);
            doc.setFont('helvetica', 'bold');
            doc.text('Analysis Metrics', 20, yPos);
            yPos += 10;

            doc.setFontSize(11);
            doc.setFont('helvetica', 'normal');

            const metrics = [
                ['Number of Lesions', result.metrics.num_lesions.toString()],
                ['Total Area', `${result.metrics.total_area_mm2.toFixed(2)} mm²`],
                ['Largest Lesion', `${result.metrics.largest_area_mm2.toFixed(2)} mm²`],
                ['Total Perimeter', `${result.metrics.total_perimeter_mm.toFixed(2)} mm`],
                ['Analysis Threshold', threshold.toFixed(2)],
                ['Pixel Spacing', `${pixelSpacing} mm`],
                ['Inference Time', `${result.inference_time_ms.toFixed(0)} ms`],
                ['Image Size', `${result.image_size.width} × ${result.image_size.height} px`]
            ];

            metrics.forEach(([label, value]) => {
                doc.setFont('helvetica', 'normal');
                doc.text(label + ':', 25, yPos);
                doc.setFont('helvetica', 'bold');
                doc.text(value, 100, yPos);
                yPos += 8;
            });

            yPos += 10;

            // Add images if available
            if (result.overlay) {
                doc.setFontSize(14);
                doc.setFont('helvetica', 'bold');
                doc.text('Segmentation Overlay', 20, yPos);
                yPos += 5;

                const imgWidth = 170;
                const imgHeight = 120;
                doc.addImage(
                    `data:image/png;base64,${result.overlay}`,
                    'PNG',
                    20,
                    yPos,
                    imgWidth,
                    imgHeight
                );
                yPos += imgHeight + 15;
            }

            // Lesion Details
            if (result.metrics.centroids && result.metrics.centroids.length > 0) {
                if (yPos > 250) {
                    doc.addPage();
                    yPos = 20;
                }

                doc.setFontSize(14);
                doc.setFont('helvetica', 'bold');
                doc.text('Lesion Locations', 20, yPos);
                yPos += 10;

                doc.setFontSize(10);
                doc.setFont('helvetica', 'normal');

                result.metrics.centroids.forEach((centroid, idx) => {
                    doc.text(`Lesion ${idx + 1}: X=${centroid.x}px, Y=${centroid.y}px`, 25, yPos);
                    yPos += 6;
                });
            }

            // Footer
            const pageCount = doc.internal.pages.length - 1;
            doc.setFontSize(8);
            doc.setTextColor(128, 128, 128);
            for (let i = 1; i <= pageCount; i++) {
                doc.setPage(i);
                doc.text(
                    'This report is generated by DeepBreast AI for clinical decision support. Always consult with a qualified radiologist.',
                    pageWidth / 2,
                    doc.internal.pageSize.getHeight() - 10,
                    { align: 'center' }
                );
            }

            // Save PDF
            doc.save(`segmentation_report_${new Date().toISOString().split('T')[0]}.pdf`);

        } catch (err) {
            console.error('PDF generation failed:', err);
            setError('Failed to generate PDF report');
        } finally {
            setIsGeneratingPDF(false);
        }
    };

    // Draw contours on canvas
    useEffect(() => {
        if (!result?.contours || !canvasRef.current || !showContours) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Set canvas size to match image
        canvas.width = result.image_size.width;
        canvas.height = result.image_size.height;

        // Draw each contour
        result.contours.forEach((contour) => {
            if (contour.length < 2) return;

            ctx.beginPath();
            ctx.moveTo(contour[0][0], contour[0][1]);

            contour.forEach(([x, y]) => {
                ctx.lineTo(x, y);
            });

            ctx.closePath();

            // Animated glow effect
            if (contourAnimated) {
                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = 3;
                ctx.shadowColor = '#22c55e';
                ctx.shadowBlur = 10;
            } else {
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 2;
                ctx.shadowBlur = 0;
            }

            ctx.stroke();
        });
    }, [result?.contours, showContours, contourAnimated]);

    const getDisplayImage = () => {
        if (!result) return previewUrl;
        switch (viewMode) {
            case 'overlay':
                return `data:image/png;base64,${result.overlay}`;
            case 'mask':
                return `data:image/png;base64,${result.mask}`;
            case 'original':
                // For TIFF files, previewUrl is null, so use result.original from API
                return previewUrl || `data:image/png;base64,${result.original}`;
            default:
                return previewUrl || `data:image/png;base64,${result.original}`;
        }
    };


    return (
        <div className="segmentation-page">
            <div className="page-header">
                <div className="header-content">
                    <div className="title-section">
                        <Target className="page-icon" />
                        <div>
                            <h1>Tumor Segmentation</h1>
                            <p>U-Net based pixel-level tumor detection</p>
                        </div>
                    </div>
                    <div className="header-badges">
                        <span className="badge badge-unet">
                            <Layers size={14} />
                            Attention U-Net
                        </span>
                        <span className="badge badge-ai">
                            <Activity size={14} />
                            Deep Learning
                        </span>
                    </div>
                </div>
            </div>

            <div className="segmentation-container">
                {/* Left Panel - Image Upload & Display */}
                <div className="image-panel">
                    <div className="panel-header">
                        <h3><Upload size={18} /> Image</h3>
                        {result && (
                            <div className="view-controls">
                                <button
                                    className={`view-btn ${viewMode === 'overlay' ? 'active' : ''}`}
                                    onClick={() => setViewMode('overlay')}
                                    title="Show Overlay"
                                >
                                    <Layers size={16} />
                                </button>
                                <button
                                    className={`view-btn ${viewMode === 'mask' ? 'active' : ''}`}
                                    onClick={() => setViewMode('mask')}
                                    title="Show Mask Only"
                                >
                                    <Circle size={16} />
                                </button>
                                <button
                                    className={`view-btn ${viewMode === 'original' ? 'active' : ''}`}
                                    onClick={() => setViewMode('original')}
                                    title="Show Original"
                                >
                                    <Square size={16} />
                                </button>
                                <button
                                    className={`view-btn ${viewMode === 'compare' ? 'active' : ''}`}
                                    onClick={() => setViewMode('compare')}
                                    title="Before/After Compare"
                                >
                                    <GripVertical size={16} />
                                </button>
                            </div>
                        )}
                    </div>

                    <div
                        className={`upload-area ${(previewUrl || selectedFile) ? 'has-image' : ''}`}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onClick={() => !previewUrl && !selectedFile && fileInputRef.current?.click()}
                    >
                        {/* Show result overlay/mask if available */}
                        {result ? (
                            viewMode === 'compare' ? (
                                /* Before/After Compare View */
                                <div className="compare-container" ref={compareRef}>
                                    <div
                                        className="compare-before"
                                        style={{ clipPath: `inset(0 ${100 - comparePosition}% 0 0)` }}
                                    >
                                        <img src={`data:image/png;base64,${result.original}`} alt="Before (Original)" />
                                    </div>
                                    <div
                                        className="compare-after"
                                        style={{ clipPath: `inset(0 0 0 ${comparePosition}%)` }}
                                    >
                                        <img src={`data:image/png;base64,${result.overlay}`} alt="After (Segmentation)" />
                                    </div>
                                    <div
                                        className="compare-slider"
                                        style={{ left: `${comparePosition}%` }}
                                    >
                                        <input
                                            type="range"
                                            min="0"
                                            max="100"
                                            value={comparePosition}
                                            onChange={(e) => setComparePosition(parseInt(e.target.value))}
                                            className="compare-range"
                                        />
                                        <div className="compare-handle">
                                            <GripVertical size={20} />
                                        </div>
                                    </div>
                                    {result?.metrics.tumor_detected && (
                                        <div className="tumor-indicator animated">
                                            <AlertCircle size={16} />
                                            Tumor Detected
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="image-preview-wrapper">
                                    {/* View Mode Toggle */}
                                    <div className="image-view-mode-toggle">
                                        <button
                                            className={`view-mode-btn ${imageViewMode === 'simple' ? 'active' : ''}`}
                                            onClick={(e) => { e.stopPropagation(); setImageViewMode('simple'); }}
                                            title="Simple View"
                                        >
                                            <ImageIcon size={14} />
                                            Simple
                                        </button>
                                        <button
                                            className={`view-mode-btn ${imageViewMode === 'zoom' ? 'active' : ''}`}
                                            onClick={(e) => { e.stopPropagation(); setImageViewMode('zoom'); }}
                                            title="Zoom / Pan"
                                        >
                                            <Eye size={14} />
                                            Zoom/Pan
                                        </button>
                                    </div>

                                    {/* Image Display based on View Mode */}
                                    {imageViewMode === 'simple' ? (
                                        <div
                                            className={`image-preview ${measurementMode !== 'none' ? 'measuring' : ''}`}
                                            ref={imageContainerRef}
                                            onClick={handleMeasurementClick}
                                        >
                                            <img
                                                src={getDisplayImage() || `data:image/png;base64,${result.overlay}`}
                                                alt="Preview"
                                                style={viewMode === 'overlay' ? { filter: `saturate(1.2)` } : {}}
                                            />

                                            {/* Contour Canvas Overlay */}
                                            {showContours && result?.contours && (
                                                <canvas
                                                    ref={canvasRef}
                                                    className={`contour-canvas ${contourAnimated ? 'animated' : ''}`}
                                                />
                                            )}

                                            {/* Measurements SVG Overlay */}
                                            {(measurements.length > 0 || currentMeasurement.length > 0) && (
                                                <svg className="measurements-overlay">
                                                    {/* Completed measurements */}
                                                    {measurements.map((m, idx) => (
                                                        <g key={idx} className="measurement-group">
                                                            {m.type === 'distance' && m.points.length === 2 && (
                                                                <>
                                                                    <line
                                                                        x1={m.points[0].x}
                                                                        y1={m.points[0].y}
                                                                        x2={m.points[1].x}
                                                                        y2={m.points[1].y}
                                                                        className="measurement-line"
                                                                    />
                                                                    <circle cx={m.points[0].x} cy={m.points[0].y} r="4" className="measurement-point" />
                                                                    <circle cx={m.points[1].x} cy={m.points[1].y} r="4" className="measurement-point" />
                                                                    <text
                                                                        x={(m.points[0].x + m.points[1].x) / 2}
                                                                        y={(m.points[0].y + m.points[1].y) / 2 - 8}
                                                                        className="measurement-label"
                                                                    >
                                                                        {m.value.toFixed(2)} mm
                                                                    </text>
                                                                </>
                                                            )}
                                                            {m.type === 'area' && (
                                                                <>
                                                                    <polygon
                                                                        points={m.points.map(p => `${p.x},${p.y}`).join(' ')}
                                                                        className="measurement-area"
                                                                    />
                                                                    {m.points.map((p, i) => (
                                                                        <circle key={i} cx={p.x} cy={p.y} r="3" className="measurement-point" />
                                                                    ))}
                                                                    <text
                                                                        x={m.points.reduce((s, p) => s + p.x, 0) / m.points.length}
                                                                        y={m.points.reduce((s, p) => s + p.y, 0) / m.points.length}
                                                                        className="measurement-label"
                                                                    >
                                                                        {m.value.toFixed(2)} mm²
                                                                    </text>
                                                                </>
                                                            )}
                                                        </g>
                                                    ))}

                                                    {/* Current measurement in progress */}
                                                    {currentMeasurement.length > 0 && (
                                                        <g className="current-measurement">
                                                            {currentMeasurement.map((p, i) => (
                                                                <React.Fragment key={i}>
                                                                    <circle cx={p.x} cy={p.y} r="5" className="measurement-point active" />
                                                                    {i > 0 && (
                                                                        <line
                                                                            x1={currentMeasurement[i - 1].x}
                                                                            y1={currentMeasurement[i - 1].y}
                                                                            x2={p.x}
                                                                            y2={p.y}
                                                                            className="measurement-line current"
                                                                        />
                                                                    )}
                                                                </React.Fragment>
                                                            ))}
                                                        </g>
                                                    )}
                                                </svg>
                                            )}

                                            {/* Measurement Mode Indicator */}
                                            {measurementMode !== 'none' && (
                                                <div className="measurement-mode-indicator">
                                                    <Ruler size={14} />
                                                    {measurementMode === 'distance' ? 'Click two points to measure' : 'Click points, double-click to finish'}
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="image-viewer-container-wrapper">
                                            <ImageViewer
                                                src={getDisplayImage() || `data:image/png;base64,${result.overlay}`}
                                                alt="Segmentation Result"
                                            />
                                        </div>
                                    )}

                                    {/* Overlays - Show on both modes but only in simple */}
                                    {imageViewMode === 'simple' && (
                                        <>
                                            {result?.metrics.tumor_detected && viewMode === 'overlay' && (
                                                <div className="tumor-indicator animated">
                                                    <AlertCircle size={16} />
                                                    Tumor Detected
                                                </div>
                                            )}
                                            {viewMode === 'mask' && (
                                                <div className="mask-legend">
                                                    <span className="legend-item tumor">● Tumor Region</span>
                                                </div>
                                            )}
                                            {viewMode === 'overlay' && (
                                                <div className="mask-legend heatmap">
                                                    <div className="legend-row">
                                                        <span className="dot red"></span> High Confidence
                                                    </div>
                                                    <div className="legend-row">
                                                        <span className="dot yellow"></span> Medium Confidence
                                                    </div>
                                                    <div className="legend-row">
                                                        <span className="dot blue"></span> Edge / Low Conf.
                                                    </div>
                                                </div>
                                            )}
                                        </>
                                    )}
                                </div>
                            )
                        ) : previewUrl ? (
                            <div className="image-preview">
                                <img src={previewUrl} alt="Preview" />
                            </div>
                        ) : selectedFile ? (
                            /* TIFF/DICOM file selected but can't preview */
                            <div className="tiff-placeholder">
                                <Layers size={48} />
                                <p className="tiff-filename">{selectedFile.name}</p>
                                <span className="tiff-info">
                                    {selectedFile.name.toLowerCase().endsWith('.dcm') || selectedFile.name.toLowerCase().endsWith('.dicom')
                                        ? '🏥 DICOM format - Preview after analysis'
                                        : 'TIFF format - Preview after analysis'}
                                </span>
                                <span className="tiff-size">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                            </div>
                        ) : (
                            <div className="upload-placeholder">
                                <Upload size={48} />
                                <p>Drop mammography image here</p>
                                <span>or click to browse</span>
                                <span className="format-hint">Supports: PNG, JPG, TIFF, DICOM</span>
                            </div>
                        )}
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/*,.tif,.tiff,.dcm,.dicom"
                            onChange={handleFileSelect}
                            style={{ display: 'none' }}
                        />
                    </div>


                    {(previewUrl || selectedFile) && (
                        <div className="image-actions">
                            <button className="btn btn-secondary" onClick={handleReset}>
                                <RefreshCw size={16} />
                                Reset
                            </button>
                            {result && (
                                <>
                                    {/* Measurement Tools */}
                                    <div className="measurement-tools">
                                        <button
                                            className={`btn btn-tool ${measurementMode === 'distance' ? 'active' : ''}`}
                                            onClick={() => setMeasurementMode(measurementMode === 'distance' ? 'none' : 'distance')}
                                            title="Measure Distance"
                                        >
                                            <Ruler size={16} />
                                        </button>
                                        <button
                                            className={`btn btn-tool ${measurementMode === 'area' ? 'active' : ''}`}
                                            onClick={() => setMeasurementMode(measurementMode === 'area' ? 'none' : 'area')}
                                            title="Measure Area"
                                        >
                                            <Move size={16} />
                                        </button>
                                        {measurements.length > 0 && (
                                            <button
                                                className="btn btn-tool"
                                                onClick={clearMeasurements}
                                                title="Clear Measurements"
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        )}
                                    </div>

                                    {/* Contour Controls */}
                                    <div className="contour-controls">
                                        <button
                                            className={`btn btn-tool ${showContours ? 'active' : ''}`}
                                            onClick={() => setShowContours(!showContours)}
                                            title="Toggle Contours"
                                        >
                                            <Crosshair size={16} />
                                        </button>
                                        {showContours && (
                                            <button
                                                className={`btn btn-tool ${contourAnimated ? 'active' : ''}`}
                                                onClick={() => setContourAnimated(!contourAnimated)}
                                                title="Animate Contours"
                                            >
                                                <Activity size={16} />
                                            </button>
                                        )}
                                    </div>

                                    <button className="btn btn-secondary" onClick={handleDownloadMask}>
                                        <ArrowDownToLine size={16} />
                                        Mask
                                    </button>
                                    <button className="btn btn-secondary" onClick={handleDownloadOverlay}>
                                        <Download size={16} />
                                        Overlay
                                    </button>
                                    <button
                                        className="btn btn-primary btn-pdf"
                                        onClick={handleGeneratePDF}
                                        disabled={isGeneratingPDF}
                                    >
                                        {isGeneratingPDF ? (
                                            <Loader2 size={16} className="spin" />
                                        ) : (
                                            <FileText size={16} />
                                        )}
                                        PDF Report
                                    </button>
                                </>
                            )}
                        </div>
                    )}
                </div>

                {/* Right Panel - Controls & Results */}
                <div className="controls-panel">
                    {/* Settings Card */}
                    <div className="settings-card">
                        <h3><Activity size={18} /> Settings</h3>

                        <div className="setting-group">
                            <label>Segmentation Threshold</label>
                            <div className="slider-container">
                                <input
                                    type="range"
                                    min="0.1"
                                    max="0.7"
                                    step="0.05"
                                    value={threshold}
                                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                />
                                <span className="slider-value">{threshold.toFixed(2)}</span>
                            </div>
                            <p className="setting-hint">Recommended: 0.30 (optimized for best results)</p>
                        </div>

                        <div className="setting-group">
                            <label>Pixel Spacing (mm)</label>
                            <input
                                type="number"
                                className="number-input"
                                min="0.01"
                                max="10"
                                step="0.01"
                                value={pixelSpacing}
                                onChange={(e) => setPixelSpacing(parseFloat(e.target.value))}
                            />
                            <p className="setting-hint">Physical size per pixel for area calculation</p>
                        </div>

                        {result && (
                            <div className="setting-group">
                                <label><SlidersHorizontal size={14} /> Overlay Opacity</label>
                                <div className="slider-container">
                                    <input
                                        type="range"
                                        min="0.1"
                                        max="1"
                                        step="0.1"
                                        value={overlayOpacity}
                                        onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                                    />
                                    <span className="slider-value">{Math.round(overlayOpacity * 100)}%</span>
                                </div>
                                <p className="setting-hint">Adjust tumor highlight visibility</p>
                            </div>
                        )}

                        <button
                            className={`btn btn-primary btn-analyze ${isLoading ? 'loading' : ''}`}
                            onClick={handleSegment}
                            disabled={!selectedFile || isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="spin" size={18} />
                                    Segmenting...
                                </>
                            ) : (
                                <>
                                    <Scan size={18} />
                                    Segment Tumor
                                </>
                            )}
                        </button>
                    </div>

                    {/* Error Display */}
                    {error && (
                        <div className="error-card">
                            <AlertCircle size={20} />
                            <p>{error}</p>
                        </div>
                    )}

                    {/* Results Card */}
                    {result && (
                        <div className="results-card">
                            <h3><CheckCircle size={18} /> Analysis Results</h3>

                            <div className={`status-badge ${result.metrics.tumor_detected ? 'detected' : 'clear'}`}>
                                {result.metrics.tumor_detected ? (
                                    <>
                                        <AlertCircle size={20} />
                                        <span>Tumor Detected</span>
                                    </>
                                ) : (
                                    <>
                                        <CheckCircle size={20} />
                                        <span>No Tumor Detected</span>
                                    </>
                                )}
                            </div>

                            {result.metrics.tumor_detected && (
                                <div className="metrics-grid">
                                    <div className="metric-item">
                                        <span className="metric-label">Lesions Found</span>
                                        <span className="metric-value">{result.metrics.num_lesions}</span>
                                    </div>
                                    <div className="metric-item">
                                        <span className="metric-label">Total Area</span>
                                        <span className="metric-value">{result.metrics.total_area_mm2.toFixed(1)} mm²</span>
                                    </div>
                                    <div className="metric-item">
                                        <span className="metric-label">Largest Lesion</span>
                                        <span className="metric-value">{result.metrics.largest_area_mm2.toFixed(1)} mm²</span>
                                    </div>
                                    <div className="metric-item">
                                        <span className="metric-label">Perimeter</span>
                                        <span className="metric-value">{result.metrics.total_perimeter_mm.toFixed(1)} mm</span>
                                    </div>
                                </div>
                            )}

                            <div className="inference-info">
                                <Info size={14} />
                                <span>Inference time: {result.inference_time_ms.toFixed(0)}ms</span>
                                <span className="separator">•</span>
                                <span>Size: {result.image_size.width}×{result.image_size.height}</span>
                            </div>
                        </div>
                    )}

                    {/* Info Card */}
                    <div className="info-card">
                        <h4><Info size={16} /> About Segmentation</h4>
                        <ul>
                            <li>Attention U-Net architecture</li>
                            <li>Validation Dice Score: <strong>0.36</strong></li>
                            <li>Trained on DMID dataset (mask threshold &gt; 200)</li>
                            <li>Pixel-level tumor boundary detection</li>
                            <li>Calculates precise tumor measurements</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Segmentation;
