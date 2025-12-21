import { Link, useLocation } from "react-router-dom";
import { Home, Brain, Eye, BarChart3, Info, Menu, X, Clock, ScanLine } from "lucide-react";
import { useState } from "react";

const Navbar = () => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/predict", label: "Histopathology", icon: Brain },
    { path: "/mammography", label: "Mammography", icon: ScanLine },
    { path: "/analysis", label: "Analysis", icon: Eye },
    { path: "/metrics", label: "Metrics", icon: BarChart3 },
    { path: "/history", label: "History", icon: Clock },
    { path: "/about", label: "About", icon: Info },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50">
      {/* Glass background */}
      <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-xl border-b border-white/10" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-105 transition-transform duration-300">
              <Brain className="w-6 h-6 text-slate-900" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold text-white tracking-tight">
                DeepBreast AI
              </h1>
              <p className="text-[10px] text-emerald-400 font-semibold tracking-widest uppercase -mt-0.5">
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
                    ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-white border border-emerald-500/30"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                  }`}
              >
                <Icon className={`w-4.5 h-4.5 transition-all duration-300 ${isActive(path) ? "text-emerald-400" : "group-hover:text-emerald-400"
                  }`} />
                <span>{label}</span>
              </Link>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2.5 rounded-xl bg-white/5 border border-white/10 text-white hover:bg-white/10 transition-colors"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-full left-0 right-0 bg-slate-900/95 backdrop-blur-xl border-b border-white/10">
          <div className="px-4 py-4 space-y-1">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all duration-300
                  ${isActive(path)
                    ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-white border border-emerald-500/30"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                  }`}
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
