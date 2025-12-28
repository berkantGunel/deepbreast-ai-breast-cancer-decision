import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./contexts/ThemeContext";
import { AuthProvider } from "./contexts/AuthContext";
import { ToastProvider } from "./components/Toast";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Analysis from "./pages/Analysis";
import Metrics from "./pages/Metrics";
import About from "./pages/About";
import History from "./pages/History";
import MammographyPredict from "./pages/MammographyPredict";
import Dashboard from "./pages/Dashboard";
import Comparison from "./pages/Comparison";
import Auth from "./pages/Auth";
import Patients from "./pages/Patients";
import Segmentation from "./pages/Segmentation";

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <ToastProvider>
          <BrowserRouter>
            <div className="min-h-screen w-full">
              <Navbar />
              {/* Spacer for fixed navbar */}
              <div className="h-20" />
              {/* Main content area */}
              <main className="w-full">
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/auth" element={<Auth />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/patients" element={<Patients />} />
                  <Route path="/predict" element={<Predict />} />
                  <Route path="/mammography" element={<MammographyPredict />} />
                  <Route path="/comparison" element={<Comparison />} />
                  <Route path="/analysis" element={<Analysis />} />
                  <Route path="/metrics" element={<Metrics />} />
                  <Route path="/history" element={<History />} />
                  <Route path="/segmentation" element={<Segmentation />} />
                  <Route path="/about" element={<About />} />
                </Routes>
              </main>
            </div>
          </BrowserRouter>
        </ToastProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
