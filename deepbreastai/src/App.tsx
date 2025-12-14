import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Analysis from "./pages/Analysis";
import Metrics from "./pages/Metrics";
import About from "./pages/About";
import History from "./pages/History";

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen w-full">
        <Navbar />
        {/* Spacer for fixed navbar */}
        <div className="h-20" />
        {/* Main content area */}
        <main className="w-full">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/history" element={<History />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;

