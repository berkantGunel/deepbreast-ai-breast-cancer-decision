import { useState, useRef, useCallback, useEffect } from 'react';
import {
    ZoomIn,
    ZoomOut,
    RotateCcw,
    Move,
    Maximize2,
    Grid3X3,
} from 'lucide-react';
import './ImageViewer.css';

interface ImageViewerProps {
    src: string;
    alt?: string;
    className?: string;
    onClose?: () => void;
}

const ImageViewer = ({ src, alt = 'Medical Image', className = '' }: ImageViewerProps) => {
    const [scale, setScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [showGrid, setShowGrid] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);

    const MIN_SCALE = 0.5;
    const MAX_SCALE = 5;
    const ZOOM_STEP = 0.25;

    const handleZoomIn = useCallback(() => {
        setScale((prev) => Math.min(prev + ZOOM_STEP, MAX_SCALE));
    }, []);

    const handleZoomOut = useCallback(() => {
        setScale((prev) => Math.max(prev - ZOOM_STEP, MIN_SCALE));
    }, []);

    const handleReset = useCallback(() => {
        setScale(1);
        setPosition({ x: 0, y: 0 });
    }, []);

    const handleWheel = useCallback((e: React.WheelEvent) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
        setScale((prev) => {
            const newScale = prev + delta;
            return Math.max(MIN_SCALE, Math.min(newScale, MAX_SCALE));
        });
    }, []);

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        if (e.button !== 0) return; // Only left mouse button
        setIsDragging(true);
        setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    }, [position]);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isDragging) return;
        setPosition({
            x: e.clientX - dragStart.x,
            y: e.clientY - dragStart.y,
        });
    }, [isDragging, dragStart]);

    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
    }, []);

    const handleMouseLeave = useCallback(() => {
        setIsDragging(false);
    }, []);

    // Touch event handlers for mobile
    const handleTouchStart = useCallback((e: React.TouchEvent) => {
        if (e.touches.length === 1) {
            setIsDragging(true);
            setDragStart({
                x: e.touches[0].clientX - position.x,
                y: e.touches[0].clientY - position.y,
            });
        }
    }, [position]);

    const handleTouchMove = useCallback((e: React.TouchEvent) => {
        if (!isDragging || e.touches.length !== 1) return;
        setPosition({
            x: e.touches[0].clientX - dragStart.x,
            y: e.touches[0].clientY - dragStart.y,
        });
    }, [isDragging, dragStart]);

    const handleTouchEnd = useCallback(() => {
        setIsDragging(false);
    }, []);

    const toggleFullscreen = useCallback(() => {
        if (!containerRef.current) return;

        if (!isFullscreen) {
            if (containerRef.current.requestFullscreen) {
                containerRef.current.requestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    }, [isFullscreen]);

    useEffect(() => {
        const handleFullscreenChange = () => {
            setIsFullscreen(!!document.fullscreenElement);
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);
        return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
    }, []);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === '+' || e.key === '=') handleZoomIn();
            if (e.key === '-') handleZoomOut();
            if (e.key === '0') handleReset();
            if (e.key === 'g') setShowGrid((prev) => !prev);
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleZoomIn, handleZoomOut, handleReset]);

    return (
        <div
            ref={containerRef}
            className={`image-viewer-container ${className} ${isFullscreen ? 'fullscreen' : ''}`}
        >
            {/* Toolbar */}
            <div className="image-viewer-toolbar">
                <div className="toolbar-group">
                    <button
                        onClick={handleZoomOut}
                        className="toolbar-btn"
                        title="Zoom Out (-)"
                        disabled={scale <= MIN_SCALE}
                    >
                        <ZoomOut size={18} />
                    </button>
                    <span className="zoom-level">{Math.round(scale * 100)}%</span>
                    <button
                        onClick={handleZoomIn}
                        className="toolbar-btn"
                        title="Zoom In (+)"
                        disabled={scale >= MAX_SCALE}
                    >
                        <ZoomIn size={18} />
                    </button>
                </div>

                <div className="toolbar-separator" />

                <div className="toolbar-group">
                    <button onClick={handleReset} className="toolbar-btn" title="Reset View (0)">
                        <RotateCcw size={18} />
                    </button>
                    <button
                        onClick={() => setShowGrid((prev) => !prev)}
                        className={`toolbar-btn ${showGrid ? 'active' : ''}`}
                        title="Toggle Grid (G)"
                    >
                        <Grid3X3 size={18} />
                    </button>
                    <button
                        onClick={toggleFullscreen}
                        className="toolbar-btn"
                        title="Fullscreen"
                    >
                        <Maximize2 size={18} />
                    </button>
                </div>

                <div className="toolbar-hint">
                    <Move size={14} /> Drag to pan
                </div>
            </div>

            {/* Image Canvas */}
            <div
                className={`image-viewer-canvas ${isDragging ? 'dragging' : ''}`}
                onWheel={handleWheel}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
            >
                {showGrid && <div className="image-grid-overlay" />}
                <img
                    ref={imageRef}
                    src={src}
                    alt={alt}
                    className="viewer-image"
                    style={{
                        transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
                        cursor: isDragging ? 'grabbing' : 'grab',
                    }}
                    draggable={false}
                />
            </div>

            {/* Quick Zoom Buttons */}
            <div className="quick-zoom-buttons">
                {[0.5, 1, 1.5, 2, 3].map((zoomLevel) => (
                    <button
                        key={zoomLevel}
                        className={`quick-zoom-btn ${Math.abs(scale - zoomLevel) < 0.1 ? 'active' : ''}`}
                        onClick={() => setScale(zoomLevel)}
                    >
                        {zoomLevel * 100}%
                    </button>
                ))}
            </div>
        </div>
    );
};

export default ImageViewer;
