import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
    Mail,
    Lock,
    User,
    Eye,
    EyeOff,
    AlertCircle,
    Loader2,
    LogIn,
    UserPlus,
    ArrowRight,
    Shield
} from 'lucide-react';
import './Auth.css';

type AuthMode = 'login' | 'register';

const Auth = () => {
    const navigate = useNavigate();
    const { login, register, isAuthenticated, isLoading } = useAuth();

    const [mode, setMode] = useState<AuthMode>('login');
    const [email, setEmail] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [fullName, setFullName] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    // Redirect if already authenticated
    useEffect(() => {
        if (!isLoading && isAuthenticated) {
            navigate('/dashboard');
        }
    }, [isAuthenticated, isLoading, navigate]);

    // Show loading while checking auth state
    if (isLoading) {
        return (
            <div className="auth-page">
                <div className="auth-loading">
                    <Loader2 size={48} className="spinner" />
                    <p>Loading...</p>
                </div>
            </div>
        );
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            if (mode === 'login') {
                await login(email, password);
                navigate('/dashboard');
            } else {
                // Validate registration
                if (password !== confirmPassword) {
                    throw new Error('Passwords do not match');
                }
                if (password.length < 6) {
                    throw new Error('Password must be at least 6 characters');
                }
                if (username.length < 3) {
                    throw new Error('Username must be at least 3 characters');
                }

                await register(email, username, password, fullName || undefined);
                navigate('/dashboard');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    const toggleMode = () => {
        setMode(mode === 'login' ? 'register' : 'login');
        setError('');
        setPassword('');
        setConfirmPassword('');
    };

    return (
        <div className="auth-page">
            {/* Background Effects */}
            <div className="auth-bg-effects">
                <div className="glow-orb orb-1" />
                <div className="glow-orb orb-2" />
                <div className="glow-orb orb-3" />
            </div>

            <div className="auth-container">
                {/* Left Side - Branding */}
                <div className="auth-brand">
                    <div className="brand-content">
                        <div className="brand-logo">
                            <Shield size={48} />
                        </div>
                        <h1>DeepBreast AI</h1>
                        <p className="brand-tagline">
                            Advanced AI-Powered Breast Cancer Detection Platform
                        </p>

                        <div className="brand-features">
                            <div className="feature">
                                <div className="feature-icon">🔬</div>
                                <div>
                                    <h4>AI Analysis</h4>
                                    <p>95.4% accuracy detection</p>
                                </div>
                            </div>
                            <div className="feature">
                                <div className="feature-icon">🩻</div>
                                <div>
                                    <h4>Mammography</h4>
                                    <p>BI-RADS classification</p>
                                </div>
                            </div>
                            <div className="feature">
                                <div className="feature-icon">👥</div>
                                <div>
                                    <h4>Patient Management</h4>
                                    <p>Track all analyses</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Side - Form */}
                <div className="auth-form-section">
                    <div className="auth-form-container">
                        {/* Mode Toggle */}
                        <div className="auth-tabs">
                            <button
                                className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
                                onClick={() => setMode('login')}
                            >
                                <LogIn size={18} />
                                Sign In
                            </button>
                            <button
                                className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
                                onClick={() => setMode('register')}
                            >
                                <UserPlus size={18} />
                                Sign Up
                            </button>
                        </div>

                        <div className="auth-form-header">
                            <h2>{mode === 'login' ? 'Welcome Back' : 'Create Account'}</h2>
                            <p>
                                {mode === 'login'
                                    ? 'Sign in to continue to your dashboard'
                                    : 'Join us to start managing patient analyses'}
                            </p>
                        </div>

                        {error && (
                            <div className="auth-error">
                                <AlertCircle size={18} />
                                <span>{error}</span>
                            </div>
                        )}

                        <form onSubmit={handleSubmit} className="auth-form">
                            {mode === 'register' && (
                                <>
                                    <div className="form-group">
                                        <label htmlFor="fullName">Full Name</label>
                                        <div className="input-wrapper">
                                            <User size={18} className="input-icon" />
                                            <input
                                                id="fullName"
                                                type="text"
                                                value={fullName}
                                                onChange={(e) => setFullName(e.target.value)}
                                                placeholder="Dr. John Smith"
                                            />
                                        </div>
                                    </div>

                                    <div className="form-group">
                                        <label htmlFor="username">Username *</label>
                                        <div className="input-wrapper">
                                            <User size={18} className="input-icon" />
                                            <input
                                                id="username"
                                                type="text"
                                                value={username}
                                                onChange={(e) => setUsername(e.target.value)}
                                                placeholder="drsmith"
                                                required
                                            />
                                        </div>
                                    </div>
                                </>
                            )}

                            <div className="form-group">
                                <label htmlFor="email">Email Address *</label>
                                <div className="input-wrapper">
                                    <Mail size={18} className="input-icon" />
                                    <input
                                        id="email"
                                        type="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        placeholder="doctor@hospital.com"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="form-group">
                                <label htmlFor="password">Password *</label>
                                <div className="input-wrapper">
                                    <Lock size={18} className="input-icon" />
                                    <input
                                        id="password"
                                        type={showPassword ? 'text' : 'password'}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="••••••••"
                                        required
                                        minLength={6}
                                    />
                                    <button
                                        type="button"
                                        className="password-toggle"
                                        onClick={() => setShowPassword(!showPassword)}
                                    >
                                        {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                                    </button>
                                </div>
                            </div>

                            {mode === 'register' && (
                                <div className="form-group">
                                    <label htmlFor="confirmPassword">Confirm Password *</label>
                                    <div className="input-wrapper">
                                        <Lock size={18} className="input-icon" />
                                        <input
                                            id="confirmPassword"
                                            type={showPassword ? 'text' : 'password'}
                                            value={confirmPassword}
                                            onChange={(e) => setConfirmPassword(e.target.value)}
                                            placeholder="••••••••"
                                            required
                                        />
                                    </div>
                                </div>
                            )}

                            <button type="submit" className="auth-submit-btn" disabled={loading}>
                                {loading ? (
                                    <>
                                        <Loader2 size={20} className="spinner" />
                                        <span>{mode === 'login' ? 'Signing in...' : 'Creating account...'}</span>
                                    </>
                                ) : (
                                    <>
                                        <span>{mode === 'login' ? 'Sign In' : 'Create Account'}</span>
                                        <ArrowRight size={20} />
                                    </>
                                )}
                            </button>
                        </form>

                        <div className="auth-footer">
                            <p>
                                {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}
                                <button onClick={toggleMode} className="auth-switch-btn">
                                    {mode === 'login' ? 'Sign Up' : 'Sign In'}
                                </button>
                            </p>
                        </div>

                        <div className="auth-divider">
                            <span>or continue without account</span>
                        </div>

                        <Link to="/" className="guest-link">
                            <span>Continue as Guest</span>
                            <ArrowRight size={16} />
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Auth;
