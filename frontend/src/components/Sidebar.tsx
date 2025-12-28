import { Link, useLocation } from "react-router-dom";
import {
  Home,
  Brain,
  Eye,
  BarChart3,
  Info,
  Menu,
  X,
  Sparkles,
} from "lucide-react";
import { useEffect } from "react";

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const Sidebar = ({ isOpen, setIsOpen }: SidebarProps) => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  // Close sidebar on route change (mobile only)
  useEffect(() => {
    if (window.innerWidth < 1024) {
      setIsOpen(false);
    }
  }, [location.pathname, setIsOpen]);

  // Handle resize - ensure sidebar state is correct
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        // On desktop, keep sidebar open by default
        setIsOpen(true);
      } else {
        // On mobile, close sidebar by default
        setIsOpen(false);
      }
    };

    // Set initial state
    handleResize();

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [setIsOpen]);

  const navItems = [
    { path: "/", label: "Home", icon: Home, description: "Welcome & Overview" },
    {
      path: "/predict",
      label: "Prediction",
      icon: Brain,
      description: "AI Cancer Detection",
    },
    {
      path: "/analysis",
      label: "Grad-CAM",
      icon: Eye,
      description: "Explainable AI",
    },
    {
      path: "/metrics",
      label: "Metrics",
      icon: BarChart3,
      description: "Model Performance",
    },
    { path: "/about", label: "About", icon: Info, description: "Project Info" },
  ];

  return (
    <>
      {/* Toggle Button - Visible on both mobile and desktop */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-6 left-6 z-[60] p-3 bg-white rounded-xl shadow-lg border border-gray-200 hover:shadow-xl transition-all hover:scale-105"
        aria-label="Toggle Sidebar"
      >
        {isOpen ? (
          <X className="w-6 h-6 text-gray-700" />
        ) : (
          <Menu className="w-6 h-6 text-gray-700" />
        )}
      </button>

      {/* Overlay for mobile - Only on mobile when sidebar is open */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar - Toggleable on both mobile and desktop */}
      <aside
        className={`fixed left-0 top-0 h-screen w-80 bg-gradient-to-b from-slate-50 to-gray-100 border-r border-gray-200/80 flex flex-col z-50 transition-transform duration-300 shadow-xl
          ${isOpen ? "translate-x-0" : "-translate-x-full"}
        `}
      >
        {/* Logo Section */}
        <div className="p-8 border-b border-gray-200/80">
          <div className="flex items-center space-x-4">
            <div className="w-14 h-14 bg-gradient-to-br from-red-500 to-pink-600 rounded-2xl flex items-center justify-center shadow-lg shadow-red-500/25">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">DeepBreast AI</h1>
              <p className="text-sm text-gray-500 mt-0.5">Cancer Detection</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-6 space-y-2 overflow-y-auto">
          <p className="px-4 py-3 text-xs font-bold text-gray-400 uppercase tracking-widest">
            Navigation
          </p>
          {navItems.map(({ path, label, icon: Icon, description }) => (
            <Link
              key={path}
              to={path}
              className={`group flex items-center space-x-4 px-4 py-4 rounded-2xl transition-all duration-300 ${
                isActive(path)
                  ? "bg-white shadow-lg shadow-gray-200/50 border border-gray-100"
                  : "hover:bg-white/70 hover:shadow-md"
              }`}
            >
              <div
                className={`p-3 rounded-xl transition-all duration-300 ${
                  isActive(path)
                    ? "bg-gradient-to-br from-red-500 to-pink-600 shadow-lg shadow-red-500/25"
                    : "bg-gray-100 group-hover:bg-gradient-to-br group-hover:from-red-500 group-hover:to-pink-600 group-hover:shadow-lg group-hover:shadow-red-500/25"
                }`}
              >
                <Icon
                  className={`w-5 h-5 transition-colors duration-300 ${
                    isActive(path)
                      ? "text-white"
                      : "text-gray-500 group-hover:text-white"
                  }`}
                />
              </div>
              <div className="flex-1 min-w-0">
                <p
                  className={`text-base font-semibold truncate transition-colors ${
                    isActive(path)
                      ? "text-gray-900"
                      : "text-gray-700 group-hover:text-gray-900"
                  }`}
                >
                  {label}
                </p>
                <p className="text-sm text-gray-400 truncate mt-0.5">
                  {description}
                </p>
              </div>
              {isActive(path) && (
                <div className="w-2 h-2 rounded-full bg-gradient-to-r from-red-500 to-pink-600 animate-pulse" />
              )}
            </Link>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200/80">
          <div className="bg-gradient-to-r from-red-500 to-pink-600 rounded-2xl p-5 text-white shadow-lg shadow-red-500/25">
            <div className="flex items-center space-x-2 mb-2">
              <Sparkles className="w-5 h-5" />
              <p className="font-bold">Graduation Project</p>
            </div>
            <p className="text-sm text-white/80">
              AI-Powered Medical Imaging Analysis
            </p>
          </div>
          <div className="flex items-center justify-center space-x-2 mt-5 text-sm text-gray-400">
            <span className="px-2.5 py-1 bg-gray-200 rounded-full text-xs font-medium text-gray-600">
              v2.0.0
            </span>
            <span>•</span>
            <span>React + FastAPI</span>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
