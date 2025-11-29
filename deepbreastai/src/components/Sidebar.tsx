import { Link, useLocation } from "react-router-dom";
import { Home, Brain, Eye, BarChart3, Info, Menu, X } from "lucide-react";
import { useEffect } from "react";

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const Sidebar = ({ isOpen, setIsOpen }: SidebarProps) => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  // Close sidebar on route change (mobile)
  useEffect(() => {
    if (window.innerWidth < 1024) {
      setIsOpen(false);
    }
  }, [location.pathname, setIsOpen]);

  const navItems = [
    { path: "/", label: "Home", icon: Home, description: "Welcome page" },
    {
      path: "/predict",
      label: "Prediction",
      icon: Brain,
      description: "AI diagnosis",
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
      description: "Performance",
    },
    { path: "/about", label: "About", icon: Info, description: "Project info" },
  ];

  return (
    <>
      {/* Mobile Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="lg:hidden fixed top-4 left-4 z-[60] p-2 bg-white rounded-lg shadow-lg border border-gray-200"
      >
        {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 h-screen w-72 bg-[#f0f2f6] border-r border-gray-200 flex flex-col z-50 transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        {/* Logo Section */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-pink-600 rounded-lg flex items-center justify-center shadow-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900">DeepBreast AI</h1>
              <p className="text-xs text-gray-500">Cancer Detection</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          <p className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Navigation
          </p>
          {navItems.map(({ path, label, icon: Icon, description }) => (
            <Link
              key={path}
              to={path}
              className={`group flex items-center space-x-3 px-3 py-3 rounded-lg transition-all duration-200 ${
                isActive(path)
                  ? "bg-white shadow-sm border border-gray-200 text-red-600"
                  : "text-gray-700 hover:bg-white hover:shadow-sm"
              }`}
            >
              <div
                className={`p-2 rounded-md transition-colors ${
                  isActive(path)
                    ? "bg-red-50"
                    : "bg-gray-100 group-hover:bg-red-50"
                }`}
              >
                <Icon
                  className={`w-4 h-4 ${
                    isActive(path)
                      ? "text-red-600"
                      : "text-gray-600 group-hover:text-red-600"
                  }`}
                />
              </div>
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm font-medium truncate ${
                    isActive(path) ? "text-red-600" : "text-gray-900"
                  }`}
                >
                  {label}
                </p>
                <p className="text-xs text-gray-500 truncate">{description}</p>
              </div>
            </Link>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200">
          <div className="bg-gradient-to-r from-red-500 to-pink-600 rounded-lg p-4 text-white">
            <p className="text-sm font-semibold mb-1">🎓 Graduation Project</p>
            <p className="text-xs opacity-90">AI-Powered Medical Imaging</p>
          </div>
          <p className="text-center text-xs text-gray-400 mt-3">
            v2.0.0 • Built with React
          </p>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
