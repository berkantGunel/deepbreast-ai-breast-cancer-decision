import { Link, useLocation } from "react-router-dom";
import {
  Home,
  Brain,
  Eye,
  BarChart3,
  Info,
  Menu,
  X,
  Clock,
  ScanLine,
  LayoutDashboard,
  GitCompare,
  Sun,
  Moon,
  Users,
  LogIn,
  LogOut,
  Target,
  ChevronDown
} from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from "../contexts/AuthContext";
import "./Navbar.css";

interface DropdownItem {
  path: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface NavItem {
  path?: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  dropdown?: DropdownItem[];
  requiresAuth?: boolean;
}

const Navbar = () => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const { toggleTheme, isDark } = useTheme();
  const { user, isAuthenticated, logout } = useAuth();
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const isActive = (path: string) => location.pathname === path;
  const isDropdownActive = (items: DropdownItem[]) => items.some(item => location.pathname === item.path);

  const navItems: NavItem[] = [
    { path: "/", label: "Home", icon: Home },
    { path: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    {
      label: "Histopathology",
      icon: Brain,
      dropdown: [
        { path: "/predict", label: "Analysis", icon: Eye },
        { path: "/comparison", label: "Compare", icon: GitCompare },
      ]
    },
    {
      label: "Mammography",
      icon: ScanLine,
      dropdown: [
        { path: "/mammography", label: "Analysis", icon: Eye },
        { path: "/segmentation", label: "Segmentation", icon: Target },
        { path: "/analysis", label: "Compare", icon: GitCompare },
      ]
    },
    { path: "/metrics", label: "Metrics", icon: BarChart3 },
    { path: "/history", label: "History", icon: Clock },
    { path: "/patients", label: "Patients", icon: Users, requiresAuth: true },
    { path: "/about", label: "About", icon: Info },
  ];

  const handleDropdownToggle = (label: string) => {
    setOpenDropdown(openDropdown === label ? null : label);
  };

  return (
    <nav className="navbar-container">
      {/* Glass background */}
      <div className="navbar-backdrop" />

      <div className="navbar-content">
        <div className="navbar-inner">
          {/* Logo */}
          <Link to="/" className="navbar-logo">
            <div className="logo-icon">
              <Brain className="w-6 h-6 text-slate-900" />
            </div>
            <div className="logo-text">
              <h1>DeepBreast AI</h1>
              <p>Medical Imaging Platform</p>
            </div>
          </Link>

          {/* Center Navigation */}
          <div className="navbar-nav" ref={dropdownRef}>
            {navItems.map((item) => {
              // Skip if requires auth and not authenticated
              if (item.requiresAuth && !isAuthenticated) return null;

              // Dropdown item
              if (item.dropdown) {
                const Icon = item.icon;
                const isDropdownOpen = openDropdown === item.label;
                const hasActiveChild = isDropdownActive(item.dropdown);

                return (
                  <div key={item.label} className="nav-dropdown-container">
                    <button
                      className={`nav-item nav-dropdown-trigger ${hasActiveChild ? 'active' : ''}`}
                      onClick={() => handleDropdownToggle(item.label)}
                    >
                      <Icon className="nav-icon" />
                      <span>{item.label}</span>
                      <ChevronDown className={`dropdown-arrow ${isDropdownOpen ? 'open' : ''}`} />
                    </button>

                    {isDropdownOpen && (
                      <div className="nav-dropdown-menu">
                        {item.dropdown.map((subItem) => {
                          const SubIcon = subItem.icon;
                          return (
                            <Link
                              key={subItem.path}
                              to={subItem.path}
                              className={`nav-dropdown-item ${isActive(subItem.path) ? 'active' : ''}`}
                              onClick={() => setOpenDropdown(null)}
                            >
                              <SubIcon className="nav-icon" />
                              <span>{subItem.label}</span>
                            </Link>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              }

              // Regular nav item
              const Icon = item.icon;
              return (
                <Link
                  key={item.path}
                  to={item.path!}
                  className={`nav-item ${isActive(item.path!) ? 'active' : ''}`}
                >
                  <Icon className="nav-icon" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Right Side - Theme & Auth */}
          <div className="navbar-actions">
            <div className="navbar-divider" />

            <button
              onClick={toggleTheme}
              className="theme-toggle"
              aria-label="Toggle theme"
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>

            {isAuthenticated ? (
              <div className="auth-section">
                <span className="user-name">
                  {user?.full_name || user?.username}
                </span>
                <button onClick={logout} className="logout-btn">
                  <span>Logout</span>
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <Link to="/auth" className="signin-btn">
                <span>Sign In</span>
                <LogIn className="w-4 h-4" />
              </Link>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="mobile-controls">
            <button
              onClick={toggleTheme}
              className="mobile-theme-toggle"
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
              className="mobile-menu-btn"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="mobile-menu">
          <div className="mobile-menu-content">
            {navItems.map((item) => {
              if (item.requiresAuth && !isAuthenticated) return null;

              // Dropdown items in mobile - show as group
              if (item.dropdown) {
                const Icon = item.icon;
                return (
                  <div key={item.label} className="mobile-menu-group">
                    <div className="mobile-menu-group-header">
                      <Icon className="w-5 h-5" />
                      <span>{item.label}</span>
                    </div>
                    <div className="mobile-menu-group-items">
                      {item.dropdown.map((subItem) => {
                        const SubIcon = subItem.icon;
                        return (
                          <Link
                            key={subItem.path}
                            to={subItem.path}
                            onClick={() => setMobileMenuOpen(false)}
                            className={`mobile-menu-item mobile-menu-subitem ${isActive(subItem.path) ? 'active' : ''}`}
                          >
                            <SubIcon className="w-4 h-4" />
                            <span>{subItem.label}</span>
                          </Link>
                        );
                      })}
                    </div>
                  </div>
                );
              }

              // Regular item
              const Icon = item.icon;
              return (
                <Link
                  key={item.path}
                  to={item.path!}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`mobile-menu-item ${isActive(item.path!) ? 'active' : ''}`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </Link>
              );
            })}

            {/* Mobile Auth */}
            <div className="mobile-auth">
              {isAuthenticated ? (
                <>
                  <span className="mobile-user-name">{user?.full_name || user?.username}</span>
                  <button onClick={() => { logout(); setMobileMenuOpen(false); }} className="mobile-logout-btn">
                    <LogOut className="w-5 h-5" />
                    <span>Logout</span>
                  </button>
                </>
              ) : (
                <Link
                  to="/auth"
                  onClick={() => setMobileMenuOpen(false)}
                  className="mobile-signin-btn"
                >
                  <LogIn className="w-5 h-5" />
                  <span>Sign In</span>
                </Link>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
