import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Analysis from "./pages/Analysis";
import Metrics from "./pages/Metrics";
import About from "./pages/About";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-white">
        <Sidebar isOpen={sidebarOpen} setIsOpen={setSidebarOpen} />
        <main className="lg:ml-72 min-h-screen p-6 pt-16 lg:pt-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
