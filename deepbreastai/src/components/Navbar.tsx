import { Link, useLocation } from "react-router-dom";
import { Home, Brain, Eye, BarChart3, Info, Menu, X, Clock, ScanLine, LayoutDashboard, GitCompare, Sun, Moon, Users, LogIn, LogOut, User } from "lucide-react";
import { useState } from "react";
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from "../contexts/AuthContext";

const Navbar = () => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { toggleTheme, isDark } = useTheme();
  const { user, isAuthenticated, logout } = useAuth();

  const isActive = (path: string) => location.pathname === path;

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { path: "/predict", label: "Histopathology", icon: Brain },
    { path: "/mammography", label: "Mammography", icon: ScanLine },
    { path: "/comparison", label: "Compare", icon: GitCompare },
    { path: "/analysis", label: "Analysis", icon: Eye },
    { path: "/metrics", label: "Metrics", icon: BarChart3 },
    { path: "/history", label: "History", icon: Clock },
    { path: "/patients", label: "Patients", icon: Users, requiresAuth: true },
    { path: "/about", label: "About", icon: Info },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50">
      {/* Glass background */}
      <div
        className="absolute inset-0 backdrop-blur-xl border-b"
        style={{
          background: 'var(--nav-bg)',
          borderColor: 'var(--color-border)'
        }}
      />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-105 transition-transform duration-300">
              <Brain className="w-6 h-6 text-slate-900" />
            </div>
            <div className="hidden sm:block">
              <h1 style={{ color: 'var(--color-text-primary)' }} className="text-xl font-bold tracking-tight">
                DeepBreast AI
              </h1>
              <p className="text-[10px] text-emerald-500 font-semibold tracking-widest uppercase -mt-0.5">
                Cancer Detection
              </p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`group flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium text-sm transition-all duration-300
                  ${isActive(path)
                    ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30"
                    : "hover:bg-white/5"
                  }`}
                style={{
                  color: isActive(path) ? 'var(--color-text-primary)' : 'var(--color-text-secondary)'
                }}
              >
                <Icon className={`w-4.5 h-4.5 transition-all duration-300 ${isActive(path) ? "text-emerald-400" : "group-hover:text-emerald-400"
                  }`} />
                <span>{label}</span>
              </Link>
            ))}

            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              className="ml-2 p-2.5 rounded-xl transition-all duration-300 hover:scale-105"
              style={{
                background: isDark ? 'rgba(251, 191, 36, 0.15)' : 'rgba(139, 92, 246, 0.15)',
                border: `1px solid ${isDark ? 'rgba(251, 191, 36, 0.3)' : 'rgba(139, 92, 246, 0.3)'}`
              }}
              aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
              title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            >
              {isDark ? (
                <Sun className="w-5 h-5 text-amber-400" />
              ) : (
                <Moon className="w-5 h-5 text-violet-500" />
              )}
            </button>

            {/* Auth Button */}
            {isAuthenticated ? (
              <div className="flex items-center gap-2 ml-2">
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20">
                  <User className="w-4 h-4 text-emerald-400" />
                  <span className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
                    {user?.full_name || user?.username}
                  </span>
                </div>
                <button
                  onClick={logout}
                  className="p-2.5 rounded-xl transition-all duration-300 hover:scale-105"
                  style={{
                    background: 'rgba(239, 68, 68, 0.15)',
                    border: '1px solid rgba(239, 68, 68, 0.3)'
                  }}
                  title="Logout"
                >
                  <LogOut className="w-5 h-5 text-red-400" />
                </button>
              </div>
            ) : (
              <Link
                to="/auth"
                className="ml-2 flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium text-sm transition-all duration-300 hover:scale-105"
                style={{
                  background: 'linear-gradient(135deg, #10b981, #06b6d4)',
                  color: '#0f172a'
                }}
              >
                <LogIn className="w-4 h-4" />
                <span>Sign In</span>
              </Link>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center gap-2">
            {/* Mobile Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2.5 rounded-xl transition-colors"
              style={{
                background: isDark ? 'rgba(251, 191, 36, 0.15)' : 'rgba(139, 92, 246, 0.15)',
                border: `1px solid ${isDark ? 'rgba(251, 191, 36, 0.3)' : 'rgba(139, 92, 246, 0.3)'}`
              }}
              aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            >
              {isDark ? (
                <Sun className="w-5 h-5 text-amber-400" />
              ) : (
                <Moon className="w-5 h-5 text-violet-500" />
              )}
            </button>

            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2.5 rounded-xl transition-colors"
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--color-border)',
                color: 'var(--color-text-primary)'
              }}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div
          className="md:hidden absolute top-full left-0 right-0 backdrop-blur-xl border-b"
          style={{
            background: 'var(--nav-bg)',
            borderColor: 'var(--color-border)'
          }}
        >
          <div className="px-4 py-4 space-y-1">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all duration-300
                  ${isActive(path)
                    ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30"
                    : "hover:bg-white/5"
                  }`}
                style={{
                  color: isActive(path) ? 'var(--color-text-primary)' : 'var(--color-text-secondary)'
                }}
              >
                <Icon className={`w-5 h-5 ${isActive(path) ? "text-emerald-400" : ""}`} />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;

