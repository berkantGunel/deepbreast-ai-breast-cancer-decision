import { Link, useLocation } from "react-router-dom";
import { Home, Brain, Eye, BarChart3, Info } from "lucide-react";

const Navbar = () => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/predict", label: "Predict", icon: Brain },
    { path: "/analysis", label: "Analysis", icon: Eye },
    { path: "/metrics", label: "Metrics", icon: BarChart3 },
    { path: "/about", label: "About", icon: Info },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/90 backdrop-blur-xl border-b border-white/10 shadow-2xl">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 sm:h-20">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-xl group-hover:scale-105 transition-all duration-300">
              <Brain className="w-6 h-6 text-slate-950 font-bold" />
            </div>
            <div className="hidden sm:flex flex-col">
              <h1 className="text-xl font-black text-white tracking-tight">DeepBreast AI</h1>
              <p className="text-[10px] text-emerald-400 font-medium -mt-1 tracking-wider">CANCER DETECTION</p>
            </div>
          </Link>

          <div className="flex items-center gap-2">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`group flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium transition-all duration-300 ${
                  isActive(path)
                    ? "bg-gradient-to-r from-emerald-400/20 to-cyan-400/20 text-white border border-emerald-400/30 shadow-lg shadow-emerald-500/20"
                    : "text-slate-300 hover:bg-white/5 hover:text-white border border-transparent"
                }`}
              >
                <Icon className={`w-5 h-5 transition-transform group-hover:scale-110 ${
                  isActive(path) ? "text-emerald-400" : ""
                }`} />
                <span className="hidden md:inline text-sm font-semibold">{label}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
