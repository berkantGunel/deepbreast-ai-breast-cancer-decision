import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';

interface User {
    id: number;
    email: string;
    username: string;
    full_name: string | null;
    role: string;
    is_active: boolean;
    created_at: string;
}

interface AuthContextType {
    user: User | null;
    token: string | null;
    isLoading: boolean;
    isAuthenticated: boolean;
    login: (email: string, password: string) => Promise<void>;
    register: (email: string, username: string, password: string, fullName?: string) => Promise<void>;
    logout: () => void;
    refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Load token from localStorage on mount
    useEffect(() => {
        const storedToken = localStorage.getItem('auth_token');
        const storedRefreshToken = localStorage.getItem('refresh_token');

        if (storedToken) {
            setToken(storedToken);
            // Verify token and get user data
            fetchUser(storedToken).catch(() => {
                // Token invalid, try refresh
                if (storedRefreshToken) {
                    refreshToken(storedRefreshToken);
                } else {
                    logout();
                }
            });
        } else {
            setIsLoading(false);
        }
    }, []);

    const fetchUser = async (authToken: string, shouldThrow: boolean = false) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${authToken}`,
                },
            });

            if (response.ok) {
                const userData = await response.json();
                setUser(userData);
                setToken(authToken);
            } else {
                // Clear invalid token from storage - silently logout
                localStorage.removeItem('auth_token');
                localStorage.removeItem('refresh_token');
                setToken(null);
                setUser(null);
                if (shouldThrow) {
                    throw new Error('Session expired. Please login again.');
                }
            }
        } catch (error) {
            // Network error or API error
            console.error('Auth error:', error);
            localStorage.removeItem('auth_token');
            localStorage.removeItem('refresh_token');
            setToken(null);
            setUser(null);
            if (shouldThrow) {
                throw error;
            }
        } finally {
            setIsLoading(false);
        }
    };

    const refreshToken = async (refreshTokenValue: string) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/refresh?refresh_token=${refreshTokenValue}`, {
                method: 'POST',
            });

            if (response.ok) {
                const data = await response.json();
                localStorage.setItem('auth_token', data.access_token);
                localStorage.setItem('refresh_token', data.refresh_token);
                setToken(data.access_token);
                await fetchUser(data.access_token);
            } else {
                logout();
            }
        } catch {
            logout();
        }
    };

    const login = async (email: string, password: string) => {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Login failed');
        }

        const data = await response.json();
        localStorage.setItem('auth_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        setToken(data.access_token);
        await fetchUser(data.access_token, true);
    };

    const register = async (email: string, username: string, password: string, fullName?: string) => {
        const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email,
                username,
                password,
                full_name: fullName || null
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Registration failed');
        }

        // Auto login after registration
        await login(email, password);
    };

    const logout = () => {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
        setToken(null);
        setUser(null);
        setIsLoading(false);
    };

    const refreshUser = async () => {
        if (token) {
            await fetchUser(token);
        }
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                token,
                isLoading,
                isAuthenticated: !!user,
                login,
                register,
                logout,
                refreshUser,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}
