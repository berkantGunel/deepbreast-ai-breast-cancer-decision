import { Link, useLocation } from "react-router-dom";
import { Activity, Brain, BarChart3, Info } from "lucide-react";

const Navbar = () => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  const navItems = [
    { path: "/", label: "Home", icon: Activity },
    { path: "/predict", label: "Prediction", icon: Brain },
    { path: "/analysis", label: "Analysis", icon: Activity },
    { path: "/metrics", label: "Metrics", icon: BarChart3 },
    { path: "/about", label: "About", icon: Info },
  ];

  return (
    <nav className="bg-gradient-to-r from-pink-600 to-red-600 text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <Brain className="w-8 h-8" />
            <h1 className="text-xl font-bold">DeepBreast AI</h1>
          </div>

          <div className="flex space-x-1">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                  isActive(path)
                    ? "bg-white/20 font-semibold"
                    : "hover:bg-white/10"
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
