import { useState, useEffect, createContext, useContext, useCallback } from 'react';
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react';

// Toast Types
export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
    id: string;
    message: string;
    type: ToastType;
    duration?: number;
}

interface ToastContextType {
    showToast: (message: string, type?: ToastType, duration?: number) => void;
    removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

// Custom hook to use toast
export const useToast = () => {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error('useToast must be used within a ToastProvider');
    }
    return context;
};

// Single Toast Component
const ToastItem = ({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) => {
    const [isExiting, setIsExiting] = useState(false);

    useEffect(() => {
        const duration = toast.duration || 4000;
        const timer = setTimeout(() => {
            setIsExiting(true);
            setTimeout(() => onRemove(toast.id), 300);
        }, duration);

        return () => clearTimeout(timer);
    }, [toast, onRemove]);

    const handleClose = () => {
        setIsExiting(true);
        setTimeout(() => onRemove(toast.id), 300);
    };

    const getIcon = () => {
        switch (toast.type) {
            case 'success':
                return <CheckCircle style={{ width: '20px', height: '20px', color: '#4ade80' }} />;
            case 'error':
                return <XCircle style={{ width: '20px', height: '20px', color: '#f87171' }} />;
            case 'warning':
                return <AlertTriangle style={{ width: '20px', height: '20px', color: '#fbbf24' }} />;
            case 'info':
            default:
                return <Info style={{ width: '20px', height: '20px', color: '#60a5fa' }} />;
        }
    };

    const getStyles = () => {
        const baseStyles = {
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '1rem 1.25rem',
            borderRadius: '12px',
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)',
            marginBottom: '0.75rem',
            animation: isExiting ? 'slideOut 0.3s ease forwards' : 'slideIn 0.3s ease forwards',
            backdropFilter: 'blur(12px)',
            border: '1px solid',
            maxWidth: '400px',
            minWidth: '280px',
        };

        switch (toast.type) {
            case 'success':
                return {
                    ...baseStyles,
                    background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1))',
                    borderColor: 'rgba(34, 197, 94, 0.3)',
                };
            case 'error':
                return {
                    ...baseStyles,
                    background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1))',
                    borderColor: 'rgba(239, 68, 68, 0.3)',
                };
            case 'warning':
                return {
                    ...baseStyles,
                    background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.1))',
                    borderColor: 'rgba(245, 158, 11, 0.3)',
                };
            case 'info':
            default:
                return {
                    ...baseStyles,
                    background: 'linear-gradient(135deg, rgba(96, 165, 250, 0.15), rgba(59, 130, 246, 0.1))',
                    borderColor: 'rgba(96, 165, 250, 0.3)',
                };
        }
    };

    return (
        <div style={getStyles()}>
            {getIcon()}
            <span style={{ flex: 1, color: '#f1f5f9', fontSize: '0.925rem', fontWeight: 500 }}>
                {toast.message}
            </span>
            <button
                onClick={handleClose}
                style={{
                    background: 'rgba(255, 255, 255, 0.1)',
                    border: 'none',
                    cursor: 'pointer',
                    padding: '0.35rem',
                    borderRadius: '6px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.2s',
                }}
                onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)'}
                onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'}
            >
                <X style={{ width: '14px', height: '14px', color: '#94a3b8' }} />
            </button>
        </div>
    );
};

// Toast Container Component
const ToastContainer = ({ toasts, removeToast }: { toasts: Toast[]; removeToast: (id: string) => void }) => {
    return (
        <div style={{
            position: 'fixed',
            top: '1.5rem',
            right: '1.5rem',
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-end',
        }}>
            {toasts.map((toast) => (
                <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
            ))}
            <style>{`
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                @keyframes slideOut {
                    from {
                        transform: translateX(0);
                        opacity: 1;
                    }
                    to {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                }
            `}</style>
        </div>
    );
};

// Toast Provider Component
export const ToastProvider = ({ children }: { children: React.ReactNode }) => {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const showToast = useCallback((message: string, type: ToastType = 'info', duration: number = 4000) => {
        const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const newToast: Toast = { id, message, type, duration };
        setToasts((prev) => [...prev, newToast]);
    }, []);

    const removeToast = useCallback((id: string) => {
        setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, []);

    return (
        <ToastContext.Provider value={{ showToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} removeToast={removeToast} />
        </ToastContext.Provider>
    );
};

export default ToastProvider;
