import { Brain, Upload, BarChart3, Eye, ArrowRight, Zap, Shield, Sparkles, ScanLine } from "lucide-react";
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className="sharp-page">
      {/* Hero Section */}
      <section className="sharp-section">
        <div className="sharp-card" style={{ padding: "3rem", position: "relative", overflow: "hidden" }}>
          {/* Background Decorations */}
          <div style={{
            position: "absolute",
            top: "-50%",
            right: "-10%",
            width: "400px",
            height: "400px",
            background: "rgba(139, 92, 246, 0.15)",
            borderRadius: "50%",
            filter: "blur(80px)",
            pointerEvents: "none"
          }} />
          <div style={{
            position: "absolute",
            bottom: "-30%",
            left: "-10%",
            width: "300px",
            height: "300px",
            background: "rgba(6, 182, 212, 0.15)",
            borderRadius: "50%",
            filter: "blur(80px)",
            pointerEvents: "none"
          }} />

          <div style={{ position: "relative", zIndex: 10 }}>
            {/* Status Badge */}
            <div style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.5rem",
              padding: "0.5rem 1rem",
              background: "rgba(34, 197, 94, 0.1)",
              border: "1px solid rgba(34, 197, 94, 0.3)",
              borderRadius: "20px",
              marginBottom: "1.5rem"
            }}>
              <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#22c55e", animation: "pulse 2s infinite" }} />
              <span style={{ fontSize: "0.9rem", fontWeight: 600, color: "#4ade80" }}>
                Live • Dual Model System
              </span>
            </div>

            {/* Heading */}
            <h1 style={{
              fontSize: "3rem",
              fontWeight: 700,
              background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              marginBottom: "1rem",
              lineHeight: 1.2
            }}>
              DeepBreast AI
            </h1>
            <p style={{ fontSize: "1.25rem", color: "#94a3b8", marginBottom: "2rem", maxWidth: "600px" }}>
              Precision AI for Early Breast Cancer Detection with Histopathology & Mammography Analysis
            </p>

            {/* Model Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.5rem", marginBottom: "2rem" }}>
              {/* Histopathology Card */}
              <div className="sharp-card" style={{ padding: "1.5rem", borderLeft: "3px solid #10b981" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
                  <div style={{
                    width: "48px",
                    height: "48px",
                    background: "rgba(16, 185, 129, 0.15)",
                    borderRadius: "12px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center"
                  }}>
                    <Brain style={{ width: "24px", height: "24px", color: "#10b981" }} />
                  </div>
                  <div>
                    <h3 style={{ color: "#f1f5f9", fontWeight: 600, fontSize: "1.1rem" }}>Histopathology</h3>
                    <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Microscopic tissue analysis</p>
                  </div>
                </div>
                <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
                  <div style={{ flex: 1, padding: "0.75rem", background: "rgba(16, 185, 129, 0.1)", borderRadius: "8px", textAlign: "center" }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#10b981" }}>95.4%</div>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Accuracy</div>
                  </div>
                  <div style={{ flex: 1, padding: "0.75rem", background: "rgba(6, 182, 212, 0.1)", borderRadius: "8px", textAlign: "center" }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#06b6d4" }}>277K</div>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Images</div>
                  </div>
                </div>
                <Link to="/predict" className="sharp-btn-primary" style={{ width: "100%" }}>
                  <Brain style={{ width: "18px", height: "18px" }} />
                  Analyze Histopathology
                  <ArrowRight style={{ width: "16px", height: "16px" }} />
                </Link>
              </div>

              {/* Mammography Card */}
              <div className="sharp-card" style={{ padding: "1.5rem", borderLeft: "3px solid #8b5cf6" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
                  <div style={{
                    width: "48px",
                    height: "48px",
                    background: "rgba(139, 92, 246, 0.15)",
                    borderRadius: "12px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center"
                  }}>
                    <ScanLine style={{ width: "24px", height: "24px", color: "#8b5cf6" }} />
                  </div>
                  <div>
                    <h3 style={{ color: "#f1f5f9", fontWeight: 600, fontSize: "1.1rem" }}>Mammography</h3>
                    <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>BI-RADS classification</p>
                  </div>
                </div>
                <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
                  <div style={{ flex: 1, padding: "0.75rem", background: "rgba(139, 92, 246, 0.1)", borderRadius: "8px", textAlign: "center" }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#8b5cf6" }}>67.5%</div>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Accuracy</div>
                  </div>
                  <div style={{ flex: 1, padding: "0.75rem", background: "rgba(236, 72, 153, 0.1)", borderRadius: "8px", textAlign: "center" }}>
                    <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#ec4899" }}>3</div>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Classes</div>
                  </div>
                </div>
                <Link to="/mammography" className="sharp-btn-primary" style={{ width: "100%", background: "linear-gradient(135deg, #8b5cf6, #ec4899)" }}>
                  <ScanLine style={{ width: "18px", height: "18px" }} />
                  Analyze Mammography
                  <ArrowRight style={{ width: "16px", height: "16px" }} />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="sharp-section">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem" }} className="stats-grid">
          {[
            { label: "Total Images", value: "277K+", color: "#06b6d4" },
            { label: "Models", value: "2", color: "#8b5cf6" },
            { label: "GPU Inference", value: "<2s", color: "#f59e0b" },
            { label: "Classes", value: "5", color: "#22c55e" },
          ].map((stat) => (
            <div key={stat.label} className="sharp-metric-card">
              <div className="sharp-metric-value" style={{ color: stat.color }}>{stat.value}</div>
              <div className="sharp-metric-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="sharp-section">
        <div className="sharp-header" style={{ marginBottom: "2rem" }}>
          <h1 style={{ fontSize: "2rem" }}>Key Features</h1>
          <p className="subtitle">Built for clinical workflows with transparency, speed, and rigor</p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "1.5rem" }} className="features-grid">
          {[
            {
              title: "Dual Model Analysis",
              desc: "Support for both histopathology and mammography images with specialized AI models.",
              icon: Brain,
              color: "#10b981",
            },
            {
              title: "Explainable AI (Grad-CAM)",
              desc: "Visualize which regions of the image influenced the model's decision with heatmaps.",
              icon: Eye,
              color: "#8b5cf6",
            },
            {
              title: "BI-RADS Classification",
              desc: "Mammography analysis with standardized BI-RADS categories and clinical recommendations.",
              icon: BarChart3,
              color: "#f59e0b",
            },
            {
              title: "Real-time Processing",
              desc: "GPU-accelerated inference delivers results in under 2 seconds.",
              icon: Zap,
              color: "#ec4899",
            },
          ].map((feature) => (
            <div key={feature.title} className="sharp-card" style={{ padding: "1.5rem" }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                <div style={{
                  width: "48px",
                  height: "48px",
                  background: `${feature.color}20`,
                  borderRadius: "12px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0
                }}>
                  <feature.icon style={{ width: "24px", height: "24px", color: feature.color }} />
                </div>
                <div>
                  <h3 style={{ color: "#f1f5f9", fontWeight: 600, fontSize: "1.1rem", marginBottom: "0.5rem" }}>
                    {feature.title}
                  </h3>
                  <p style={{ color: "#94a3b8", fontSize: "0.9rem", lineHeight: 1.6, margin: 0 }}>
                    {feature.desc}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="sharp-section">
        <div className="sharp-header" style={{ marginBottom: "2rem" }}>
          <h1 style={{ fontSize: "2rem" }}>How It Works</h1>
          <p className="subtitle">Simple 3-step process for instant analysis</p>
        </div>

        <div className="sharp-card" style={{ padding: "2.5rem" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "2rem" }} className="steps-grid">
            {[
              { step: "1", title: "Upload Image", desc: "Select a histopathology or mammography image", gradient: "linear-gradient(135deg, #10b981, #06b6d4)" },
              { step: "2", title: "AI Analysis", desc: "Our specialized model processes the image", gradient: "linear-gradient(135deg, #06b6d4, #8b5cf6)" },
              { step: "3", title: "Get Results", desc: "View diagnosis, confidence, and recommendations", gradient: "linear-gradient(135deg, #8b5cf6, #ec4899)" },
            ].map((item) => (
              <div key={item.title} style={{ textAlign: "center" }}>
                <div style={{
                  width: "80px",
                  height: "80px",
                  background: item.gradient,
                  borderRadius: "20px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto 1.5rem",
                  fontSize: "2rem",
                  fontWeight: 700,
                  color: "white",
                  boxShadow: "0 10px 30px -10px rgba(139, 92, 246, 0.5)"
                }}>
                  {item.step}
                </div>
                <h4 style={{ color: "#f1f5f9", fontSize: "1.25rem", fontWeight: 600, marginBottom: "0.5rem" }}>
                  {item.title}
                </h4>
                <p style={{ color: "#94a3b8", margin: 0 }}>{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section>
        <div className="sharp-card" style={{
          padding: "1.5rem",
          background: "linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(239, 68, 68, 0.05))",
          borderColor: "rgba(245, 158, 11, 0.2)"
        }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
            <div style={{
              width: "48px",
              height: "48px",
              background: "rgba(245, 158, 11, 0.15)",
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0
            }}>
              <Shield style={{ width: "24px", height: "24px", color: "#f59e0b" }} />
            </div>
            <div>
              <h4 style={{ color: "#fcd34d", fontWeight: 600, marginBottom: "0.5rem" }}>Medical Disclaimer</h4>
              <p style={{ color: "#fde68a", fontSize: "0.9rem", margin: 0, lineHeight: 1.6 }}>
                This AI system is intended as a decision-support tool for medical professionals only.
                It should not be used as a sole diagnostic method. Always consult with qualified healthcare providers.
              </p>
            </div>
          </div>
        </div>
      </section>

      <style>{`
        @media (max-width: 1024px) {
          .stats-grid { grid-template-columns: repeat(2, 1fr) !important; }
          .features-grid { grid-template-columns: 1fr !important; }
          .steps-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 640px) {
          .stats-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
};

export default Home;