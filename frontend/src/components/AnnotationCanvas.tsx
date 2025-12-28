import { useState, useRef, useCallback, useEffect } from 'react';
import {
    Pencil,
    Circle,
    Square,
    MousePointer2,
    Eraser,
    Undo2,
    Redo2,
    Trash2,
    Download,
    Palette,
    Type,
    ArrowRight,
    Minus,
} from 'lucide-react';
import './AnnotationCanvas.css';

type Tool = 'select' | 'pencil' | 'line' | 'arrow' | 'rectangle' | 'circle' | 'text' | 'eraser';

interface Point {
    x: number;
    y: number;
}

interface Annotation {
    id: string;
    tool: Tool;
    points: Point[];
    color: string;
    strokeWidth: number;
    text?: string;
}

interface AnnotationCanvasProps {
    imageSrc: string;
    onAnnotationsChange?: (annotations: Annotation[]) => void;
}

const COLORS = [
    '#ef4444', // Red
    '#f59e0b', // Amber
    '#22c55e', // Green
    '#06b6d4', // Cyan
    '#8b5cf6', // Purple
    '#ec4899', // Pink
    '#ffffff', // White
    '#000000', // Black
];

const STROKE_WIDTHS = [2, 4, 6, 8];

const AnnotationCanvas = ({
    imageSrc,
    onAnnotationsChange,
}: AnnotationCanvasProps) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const imageRef = useRef<HTMLImageElement | null>(null);

    const [tool, setTool] = useState<Tool>('pencil');
    const [color, setColor] = useState('#ef4444');
    const [strokeWidth, setStrokeWidth] = useState(4);
    const [isDrawing, setIsDrawing] = useState(false);
    const [annotations, setAnnotations] = useState<Annotation[]>([]);
    const [currentAnnotation, setCurrentAnnotation] = useState<Annotation | null>(null);
    const [history, setHistory] = useState<Annotation[][]>([[]]);
    const [historyIndex, setHistoryIndex] = useState(0);
    const [showColorPicker, setShowColorPicker] = useState(false);
    const [textInput, setTextInput] = useState('');
    const [textPosition, setTextPosition] = useState<Point | null>(null);
    const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

    // Load image and set canvas size
    useEffect(() => {
        const img = new Image();
        img.onload = () => {
            imageRef.current = img;
            if (containerRef.current) {
                const containerWidth = containerRef.current.clientWidth;
                const aspectRatio = img.height / img.width;
                const width = containerWidth;
                const height = Math.min(containerWidth * aspectRatio, 500);
                setCanvasSize({ width, height });
            }
        };
        img.src = imageSrc;
    }, [imageSrc]);

    // Redraw canvas
    const redrawCanvas = useCallback(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        const img = imageRef.current;

        if (!canvas || !ctx || !img) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw image
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw all annotations
        [...annotations, currentAnnotation].forEach((annotation) => {
            if (!annotation) return;
            drawAnnotation(ctx, annotation);
        });
    }, [annotations, currentAnnotation]);

    useEffect(() => {
        if (canvasSize.width > 0) {
            redrawCanvas();
        }
    }, [canvasSize, redrawCanvas]);

    const drawAnnotation = (ctx: CanvasRenderingContext2D, annotation: Annotation) => {
        ctx.strokeStyle = annotation.color;
        ctx.fillStyle = annotation.color;
        ctx.lineWidth = annotation.strokeWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        switch (annotation.tool) {
            case 'pencil':
                if (annotation.points.length < 2) return;
                ctx.beginPath();
                ctx.moveTo(annotation.points[0].x, annotation.points[0].y);
                annotation.points.forEach((point) => ctx.lineTo(point.x, point.y));
                ctx.stroke();
                break;

            case 'line':
                if (annotation.points.length < 2) return;
                ctx.beginPath();
                ctx.moveTo(annotation.points[0].x, annotation.points[0].y);
                ctx.lineTo(annotation.points[1].x, annotation.points[1].y);
                ctx.stroke();
                break;

            case 'arrow':
                if (annotation.points.length < 2) return;
                const [start, end] = annotation.points;
                const angle = Math.atan2(end.y - start.y, end.x - start.x);
                const headLength = 15;

                ctx.beginPath();
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();

                // Arrow head
                ctx.beginPath();
                ctx.moveTo(end.x, end.y);
                ctx.lineTo(
                    end.x - headLength * Math.cos(angle - Math.PI / 6),
                    end.y - headLength * Math.sin(angle - Math.PI / 6)
                );
                ctx.lineTo(
                    end.x - headLength * Math.cos(angle + Math.PI / 6),
                    end.y - headLength * Math.sin(angle + Math.PI / 6)
                );
                ctx.closePath();
                ctx.fill();
                break;

            case 'rectangle':
                if (annotation.points.length < 2) return;
                const [p1, p2] = annotation.points;
                ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
                break;

            case 'circle':
                if (annotation.points.length < 2) return;
                const [center, edge] = annotation.points;
                const radius = Math.sqrt(
                    Math.pow(edge.x - center.x, 2) + Math.pow(edge.y - center.y, 2)
                );
                ctx.beginPath();
                ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
                ctx.stroke();
                break;

            case 'text':
                if (annotation.text && annotation.points.length > 0) {
                    ctx.font = `${annotation.strokeWidth * 4}px Inter, sans-serif`;
                    ctx.fillText(annotation.text, annotation.points[0].x, annotation.points[0].y);
                }
                break;

            case 'eraser':
                // Eraser is handled differently - during the next redraw, erased annotations are removed
                break;
        }
    };

    const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>): Point => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (tool === 'select') return;

        const pos = getMousePos(e);

        if (tool === 'text') {
            setTextPosition(pos);
            return;
        }

        setIsDrawing(true);
        const newAnnotation: Annotation = {
            id: `annotation-${Date.now()}`,
            tool,
            points: [pos],
            color,
            strokeWidth,
        };
        setCurrentAnnotation(newAnnotation);
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!isDrawing || !currentAnnotation) return;

        const pos = getMousePos(e);

        if (tool === 'pencil') {
            setCurrentAnnotation({
                ...currentAnnotation,
                points: [...currentAnnotation.points, pos],
            });
        } else {
            setCurrentAnnotation({
                ...currentAnnotation,
                points: [currentAnnotation.points[0], pos],
            });
        }

        redrawCanvas();
    };

    const handleMouseUp = () => {
        if (!isDrawing || !currentAnnotation) return;

        setIsDrawing(false);

        const newAnnotations = [...annotations, currentAnnotation];
        setAnnotations(newAnnotations);
        setCurrentAnnotation(null);

        // Update history
        const newHistory = history.slice(0, historyIndex + 1);
        newHistory.push(newAnnotations);
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);

        if (onAnnotationsChange) {
            onAnnotationsChange(newAnnotations);
        }
    };

    const handleTextSubmit = () => {
        if (!textInput || !textPosition) return;

        const textAnnotation: Annotation = {
            id: `annotation-${Date.now()}`,
            tool: 'text',
            points: [textPosition],
            color,
            strokeWidth,
            text: textInput,
        };

        const newAnnotations = [...annotations, textAnnotation];
        setAnnotations(newAnnotations);
        setTextInput('');
        setTextPosition(null);

        // Update history
        const newHistory = history.slice(0, historyIndex + 1);
        newHistory.push(newAnnotations);
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);

        if (onAnnotationsChange) {
            onAnnotationsChange(newAnnotations);
        }
    };

    const handleUndo = () => {
        if (historyIndex > 0) {
            const newIndex = historyIndex - 1;
            setHistoryIndex(newIndex);
            setAnnotations(history[newIndex]);
        }
    };

    const handleRedo = () => {
        if (historyIndex < history.length - 1) {
            const newIndex = historyIndex + 1;
            setHistoryIndex(newIndex);
            setAnnotations(history[newIndex]);
        }
    };

    const handleClear = () => {
        setAnnotations([]);
        const newHistory = history.slice(0, historyIndex + 1);
        newHistory.push([]);
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);

        if (onAnnotationsChange) {
            onAnnotationsChange([]);
        }
    };

    const handleDownload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const link = document.createElement('a');
        link.download = `annotated-image-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    };

    const tools: { id: Tool; icon: React.ReactNode; label: string }[] = [
        { id: 'select', icon: <MousePointer2 size={18} />, label: 'Select' },
        { id: 'pencil', icon: <Pencil size={18} />, label: 'Pencil' },
        { id: 'line', icon: <Minus size={18} />, label: 'Line' },
        { id: 'arrow', icon: <ArrowRight size={18} />, label: 'Arrow' },
        { id: 'rectangle', icon: <Square size={18} />, label: 'Rectangle' },
        { id: 'circle', icon: <Circle size={18} />, label: 'Circle' },
        { id: 'text', icon: <Type size={18} />, label: 'Text' },
        { id: 'eraser', icon: <Eraser size={18} />, label: 'Eraser' },
    ];

    return (
        <div className="annotation-canvas-container" ref={containerRef}>
            {/* Toolbar */}
            <div className="annotation-toolbar">
                <div className="tool-group">
                    {tools.map((t) => (
                        <button
                            key={t.id}
                            onClick={() => setTool(t.id)}
                            className={`tool-btn ${tool === t.id ? 'active' : ''}`}
                            title={t.label}
                        >
                            {t.icon}
                        </button>
                    ))}
                </div>

                <div className="toolbar-divider" />

                <div className="tool-group">
                    <div className="color-picker-wrapper">
                        <button
                            className="tool-btn color-btn"
                            onClick={() => setShowColorPicker(!showColorPicker)}
                            title="Color"
                        >
                            <Palette size={18} />
                            <span className="current-color" style={{ backgroundColor: color }} />
                        </button>
                        {showColorPicker && (
                            <div className="color-picker-popup">
                                {COLORS.map((c) => (
                                    <button
                                        key={c}
                                        className={`color-option ${color === c ? 'active' : ''}`}
                                        style={{ backgroundColor: c }}
                                        onClick={() => {
                                            setColor(c);
                                            setShowColorPicker(false);
                                        }}
                                    />
                                ))}
                            </div>
                        )}
                    </div>

                    <div className="stroke-width-picker">
                        {STROKE_WIDTHS.map((w) => (
                            <button
                                key={w}
                                className={`stroke-btn ${strokeWidth === w ? 'active' : ''}`}
                                onClick={() => setStrokeWidth(w)}
                                title={`${w}px`}
                            >
                                <span style={{ width: w * 2, height: w * 2 }} />
                            </button>
                        ))}
                    </div>
                </div>

                <div className="toolbar-divider" />

                <div className="tool-group">
                    <button
                        onClick={handleUndo}
                        className="tool-btn"
                        disabled={historyIndex <= 0}
                        title="Undo"
                    >
                        <Undo2 size={18} />
                    </button>
                    <button
                        onClick={handleRedo}
                        className="tool-btn"
                        disabled={historyIndex >= history.length - 1}
                        title="Redo"
                    >
                        <Redo2 size={18} />
                    </button>
                    <button onClick={handleClear} className="tool-btn danger" title="Clear All">
                        <Trash2 size={18} />
                    </button>
                    <button onClick={handleDownload} className="tool-btn success" title="Download">
                        <Download size={18} />
                    </button>
                </div>
            </div>

            {/* Canvas */}
            <div className="canvas-wrapper">
                <canvas
                    ref={canvasRef}
                    width={canvasSize.width}
                    height={canvasSize.height}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    className={`annotation-canvas ${tool !== 'select' ? 'drawing' : ''}`}
                />

                {/* Text Input Overlay */}
                {textPosition && (
                    <div
                        className="text-input-overlay"
                        style={{
                            left: textPosition.x,
                            top: textPosition.y,
                        }}
                    >
                        <input
                            type="text"
                            value={textInput}
                            onChange={(e) => setTextInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleTextSubmit()}
                            placeholder="Enter text..."
                            autoFocus
                            style={{ color }}
                        />
                        <button onClick={handleTextSubmit}>Add</button>
                        <button onClick={() => setTextPosition(null)}>Cancel</button>
                    </div>
                )}
            </div>

            {/* Annotation Count */}
            <div className="annotation-status">
                <span className="annotation-count">
                    {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
                </span>
                <span className="hotkey-hint">
                    Click and drag to draw • Ctrl+Z to undo
                </span>
            </div>
        </div>
    );
};

export default AnnotationCanvas;
