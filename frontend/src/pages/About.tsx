import {
  Brain,
  Target,
  Zap,
  Shield,
  Users,
  Award,
  Github,
  Mail,
  GraduationCap,
  AlertTriangle,
  ScanLine,
} from "lucide-react";

const About = () => {
  const features = [
    { icon: Brain, title: "Deep Learning", description: "Advanced CNN architectures trained on 277,000+ histopathology images and mammography datasets.", color: "#06b6d4" },
    { icon: Target, title: "High Accuracy", description: "95.4% histopathology accuracy and 81.4% mammography accuracy with BI-RADS classification.", color: "#10b981" },
    { icon: Zap, title: "Real-time Analysis", description: "Lightning-fast GPU inference with results in under 2 seconds.", color: "#f59e0b" },
    { icon: Shield, title: "Explainable AI", description: "Grad-CAM visualizations show which tissue regions influenced the diagnosis.", color: "#8b5cf6" },
    { icon: Users, title: "Clinical Support", description: "Decision support tool for pathologists, augmenting human expertise with AI.", color: "#ec4899" },
    { icon: Award, title: "Research-Grade", description: "Built on peer-reviewed methodologies and validated medical imaging datasets.", color: "#6366f1" },
  ];

  const models = [
    {
      name: "Histopathology Model",
      type: "ResNet18",
      specs: [
        { label: "Architecture", value: "ResNet18 with custom head" },
        { label: "Input Size", value: "50x50 RGB patches" },
        { label: "Output", value: "Binary (Benign/Malignant)" },
        { label: "Dataset", value: "277,524 images" },
        { label: "Test Accuracy", value: "95.4%" },
      ],
      color: "#10b981"
    },
    {
      name: "Mammography Model",
      type: "Ensemble ML (RF + GB)",
      specs: [
        { label: "Architecture", value: "Random Forest + Gradient Boosting" },
        { label: "Input Features", value: "78 (Texture, Morphology)" },
        { label: "Output", value: "3-class BI-RADS" },
        { label: "Dataset", value: "DMID Dataset" },
        { label: "Test Accuracy", value: "81.37%" },
      ],
      color: "#8b5cf6"
    }
  ];

  return (
    <div className="sharp-page">
      {/* Header */}
      <div className="sharp-header">
        <h1>About DeepBreast AI</h1>
        <p className="subtitle">AI-Powered Breast Cancer Detection System</p>
      </div>

      {/* Project Info */}
      <div className="sharp-card" style={{ padding: "2rem", marginBottom: "2rem", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: "-50%", right: "-10%", width: "300px", height: "300px", background: "rgba(139, 92, 246, 0.1)", borderRadius: "50%", filter: "blur(60px)", pointerEvents: "none" }} />
        <div style={{ position: "relative", zIndex: 10, display: "flex", alignItems: "flex-start", gap: "1.5rem" }}>
          <div style={{ width: "64px", height: "64px", background: "rgba(16, 185, 129, 0.15)", borderRadius: "16px", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <GraduationCap style={{ width: "32px", height: "32px", color: "#10b981" }} />
          </div>
          <div>
            <h2 style={{ color: "#f1f5f9", fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.75rem" }}>Graduation Project</h2>
            <p style={{ color: "#cbd5e1", lineHeight: 1.7, marginBottom: "1rem" }}>
              This project demonstrates the application of deep learning in medical imaging for breast cancer detection.
              Combining state-of-the-art convolutional neural networks with explainable AI techniques to provide
              accurate, transparent, and reliable diagnostic support.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {["Deep Learning", "Medical AI", "Computer Vision", "Explainable AI", "BI-RADS"].map((tag) => (
                <span key={tag} style={{ padding: "0.5rem 1rem", background: "rgba(255, 255, 255, 0.05)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "8px", fontSize: "0.85rem", color: "#e2e8f0" }}>{tag}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Mission */}
      <div className="sharp-card" style={{ padding: "2rem", marginBottom: "2rem" }}>
        <h2 style={{ color: "#f1f5f9", fontSize: "1.5rem", fontWeight: 700, marginBottom: "1rem" }}>Our Mission</h2>
        <div style={{ color: "#cbd5e1", lineHeight: 1.7 }}>
          <p style={{ marginBottom: "1rem" }}>
            DeepBreast AI leverages cutting-edge deep learning technology to assist medical professionals in the early
            detection of breast cancer through histopathology and mammography image analysis.
          </p>
          <p>
            <strong style={{ color: "#f1f5f9" }}>Early detection saves lives.</strong> Our goal is to make AI-powered
            diagnostic tools accessible and trustworthy, empowering healthcare providers with advanced technology
            while maintaining the highest standards of accuracy and interpretability.
          </p>
        </div>
      </div>

      {/* Features Grid */}
      <h2 style={{ color: "#f1f5f9", fontSize: "1.5rem", fontWeight: 700, marginBottom: "1.5rem" }}>Key Features</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginBottom: "2rem" }} className="features-about-grid">
        {features.map((feature) => (
          <div key={feature.title} className="sharp-card" style={{ padding: "1.5rem" }}>
            <div style={{ width: "48px", height: "48px", background: `${feature.color}20`, borderRadius: "12px", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: "1rem" }}>
              <feature.icon style={{ width: "24px", height: "24px", color: feature.color }} />
            </div>
            <h3 style={{ color: "#f1f5f9", fontSize: "1.1rem", fontWeight: 600, marginBottom: "0.5rem" }}>{feature.title}</h3>
            <p style={{ color: "#94a3b8", fontSize: "0.9rem", lineHeight: 1.6, margin: 0 }}>{feature.description}</p>
          </div>
        ))}
      </div>

      {/* Model Specifications */}
      <h2 style={{ color: "#f1f5f9", fontSize: "1.5rem", fontWeight: 700, marginBottom: "1.5rem" }}>Model Specifications</h2>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "2rem" }} className="models-grid">
        {models.map((model) => (
          <div key={model.name} className="sharp-card" style={{ padding: "1.5rem", borderTop: `3px solid ${model.color}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.25rem" }}>
              {model.color === "#10b981" ? <Brain style={{ width: "24px", height: "24px", color: model.color }} /> : <ScanLine style={{ width: "24px", height: "24px", color: model.color }} />}
              <div>
                <h3 style={{ color: "#f1f5f9", fontWeight: 600 }}>{model.name}</h3>
                <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>{model.type}</p>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              {model.specs.map((spec) => (
                <div key={spec.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "rgba(30, 41, 59, 0.5)", borderRadius: "8px" }}>
                  <span style={{ color: "#94a3b8", fontSize: "0.9rem" }}>{spec.label}</span>
                  <span style={{ color: "#f1f5f9", fontWeight: 500, fontSize: "0.9rem" }}>{spec.value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Disclaimer */}
      <div className="sharp-card" style={{ padding: "1.5rem", marginBottom: "2rem", background: "linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(239, 68, 68, 0.05))", borderColor: "rgba(245, 158, 11, 0.2)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
          <div style={{ width: "48px", height: "48px", background: "rgba(245, 158, 11, 0.15)", borderRadius: "12px", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
            <AlertTriangle style={{ width: "24px", height: "24px", color: "#f59e0b" }} />
          </div>
          <div>
            <h4 style={{ color: "#fcd34d", fontWeight: 600, marginBottom: "0.5rem" }}>Important Disclaimer</h4>
            <p style={{ color: "#fde68a", fontSize: "0.9rem", margin: 0, lineHeight: 1.6 }}>
              This system is designed as a decision-support tool for medical professionals and should not replace
              professional medical judgment. Always consult with qualified healthcare providers for diagnosis and treatment decisions.
            </p>
          </div>
        </div>
      </div>

      {/* Contact */}
      <div className="sharp-card" style={{ padding: "2rem", textAlign: "center" }}>
        <h2 style={{ color: "#f1f5f9", fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.5rem" }}>Get in Touch</h2>
        <p style={{ color: "#94a3b8", marginBottom: "1.5rem" }}>Have questions or feedback about DeepBreast AI? We'd love to hear from you.</p>
        <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap" }}>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="sharp-btn-secondary">
            <Github style={{ width: "18px", height: "18px" }} /> GitHub
          </a>
          <a href="mailto:contact@example.com" className="sharp-btn-primary" style={{ width: "auto", display: "inline-flex" }}>
            <Mail style={{ width: "18px", height: "18px" }} /> Contact Us
          </a>
        </div>
      </div>

      {/* Footer */}
      <div style={{ textAlign: "center", marginTop: "2rem", color: "#64748b", fontSize: "0.85rem" }}>
        <p>DeepBreast AI v3.0.0 • Built with React, FastAPI & PyTorch</p>
        <p style={{ marginTop: "0.25rem" }}>© 2024 All Rights Reserved</p>
      </div>

      <style>{`
        @media (max-width: 1024px) {
          .features-about-grid { grid-template-columns: repeat(2, 1fr) !important; }
          .models-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 640px) {
          .features-about-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
};

export default About;
