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
  Database,
  Cpu,
  AlertTriangle,
  Heart,
} from "lucide-react";

const About = () => {
  const features = [
    {
      icon: Brain,
      title: "Deep Learning",
      description:
        "Advanced CNN architecture trained on 277,000+ histopathology images for exceptional accuracy.",
      accent: "from-cyan-500/20 to-blue-500/20",
    },
    {
      icon: Target,
      title: "95.4% Accuracy",
      description:
        "Validated performance demonstrating clinical-grade accuracy in cancer detection.",
      accent: "from-emerald-500/20 to-green-500/20",
    },
    {
      icon: Zap,
      title: "Real-time Analysis",
      description:
        "Lightning-fast inference with results in under 2 seconds for rapid decisions.",
      accent: "from-yellow-500/20 to-amber-500/20",
    },
    {
      icon: Shield,
      title: "Explainable AI",
      description:
        "Grad-CAM visualizations show which tissue regions influenced the diagnosis.",
      accent: "from-purple-500/20 to-violet-500/20",
    },
    {
      icon: Users,
      title: "Clinical Support",
      description:
        "Decision support tool for pathologists, augmenting human expertise with AI.",
      accent: "from-pink-500/20 to-rose-500/20",
    },
    {
      icon: Award,
      title: "Research-Grade",
      description:
        "Built on peer-reviewed methodologies and validated medical imaging datasets.",
      accent: "from-indigo-500/20 to-blue-500/20",
    },
  ];

  const techSpecs = {
    architecture: [
      { label: "Model", value: "Custom CNN with residual connections" },
      { label: "Framework", value: "PyTorch 2.7 with CUDA acceleration" },
      { label: "Input", value: "50x50 pixel RGB histopathology patches" },
      { label: "Output", value: "Binary classification (Benign/Malignant)" },
    ],
    training: [
      { label: "Dataset", value: "277,524 labeled images" },
      { label: "Validation", value: "Stratified 80/20 split" },
      { label: "Augmentation", value: "Rotation, flip, color jitter" },
      { label: "Optimization", value: "Adam optimizer with LR scheduling" },
    ],
  };

  return (
    <div className="w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto text-slate-50">
      {/* Header */}
      <section className="section animate-fade-in">
        <div className="mb-2">
          <h1 className="text-display text-white flex items-center gap-4">
            <div className="p-3 bg-purple-500/20 border border-purple-400/30 rounded-2xl">
              <Heart className="w-10 h-10 text-purple-400" />
            </div>
            About DeepBreast AI
          </h1>
        </div>
        <p className="text-body text-slate-300 mt-4 max-w-2xl">
          AI-Powered Breast Cancer Detection System for Medical Professionals
        </p>
      </section>

      {/* Project Info Card */}
      <section
        className="section animate-fade-in-up"
        style={{ animationDelay: "0.1s" }}
      >
        <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950 border border-white/10 rounded-3xl p-10 lg:p-12 text-white shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-0" />

          <div className="relative z-10 flex flex-col lg:flex-row items-start gap-8">
            <div className="p-4 bg-white/10 backdrop-blur-sm rounded-2xl border border-white/20">
              <GraduationCap className="w-12 h-12 text-emerald-400" />
            </div>
            <div className="flex-1">
              <h2 className="text-headline text-white mb-4">
                Graduation Project
              </h2>
              <p className="text-lg text-slate-200 leading-relaxed mb-6">
                This project demonstrates the application of deep learning in
                medical imaging for breast cancer detection. Combining
                state-of-the-art convolutional neural networks with explainable
                AI techniques to provide accurate, transparent, and reliable
                diagnostic support.
              </p>
              <div className="flex flex-wrap gap-3">
                <span className="badge bg-white/20 text-white border-0 px-4 py-2">
                  Deep Learning
                </span>
                <span className="badge bg-white/20 text-white border-0 px-4 py-2">
                  Medical AI
                </span>
                <span className="badge bg-white/20 text-white border-0 px-4 py-2">
                  Computer Vision
                </span>
                <span className="badge bg-white/20 text-white border-0 px-4 py-2">
                  Explainable AI
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Mission */}
      <section
        className="section animate-fade-in-up"
        style={{ animationDelay: "0.15s" }}
      >
        <div className="bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl">
          <h2 className="text-headline text-white mb-6">Our Mission</h2>
          <div className="space-y-4">
            <p className="text-body text-slate-300 leading-relaxed">
              DeepBreast AI leverages cutting-edge deep learning technology to
              assist medical professionals in the early detection of breast
              cancer through histopathology image analysis.
            </p>
            <p className="text-body text-slate-300 leading-relaxed">
              <strong className="text-white">
                Early detection saves lives.
              </strong>{" "}
              Our goal is to make AI-powered diagnostic tools accessible and
              trustworthy, empowering healthcare providers with advanced
              technology while maintaining the highest standards of accuracy and
              interpretability.
            </p>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="section">
        <h2
          className="text-headline text-white mb-8 animate-fade-in-up"
          style={{ animationDelay: "0.2s" }}
        >
          Key Features
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-gradient-to-br from-slate-900 to-slate-950 border border-white/5 rounded-2xl p-6 lg:p-8 hover:shadow-2xl hover:border-white/10 transition-all duration-300 group animate-fade-in-up"
              style={{ animationDelay: `${0.25 + index * 0.05}s` }}
            >
              <div
                className={`p-4 bg-gradient-to-br ${feature.accent} border border-white/10 rounded-2xl w-fit mb-5 transition-colors`}
              >
                <feature.icon className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-title text-white mb-3">{feature.title}</h3>
              <p className="text-body text-slate-300 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Technical Specifications */}
      <section className="section">
        <div
          className="bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl animate-fade-in-up"
          style={{ animationDelay: "0.55s" }}
        >
          <h2 className="text-headline text-white mb-8 flex items-center gap-3">
            <div className="p-2 bg-white/10 rounded-xl border border-white/20">
              <Cpu className="w-6 h-6 text-cyan-400" />
            </div>
            Technical Specifications
          </h2>
          <div className="grid md:grid-cols-2 gap-10">
            <div>
              <h3 className="text-title text-white mb-6 flex items-center gap-3">
                <Brain className="w-5 h-5 text-blue-400" />
                Model Architecture
              </h3>
              <div className="space-y-4">
                {techSpecs.architecture.map((item, index) => (
                  <div
                    key={index}
                    className="flex justify-between items-center py-4 border-b border-white/10 last:border-0"
                  >
                    <span className="text-body text-slate-300">
                      {item.label}
                    </span>
                    <span className="text-body font-semibold text-white">
                      {item.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-title text-white mb-6 flex items-center gap-3">
                <Database className="w-5 h-5 text-emerald-400" />
                Training Details
              </h3>
              <div className="space-y-4">
                {techSpecs.training.map((item, index) => (
                  <div
                    key={index}
                    className="flex justify-between items-center py-4 border-b border-white/10 last:border-0"
                  >
                    <span className="text-body text-slate-300">
                      {item.label}
                    </span>
                    <span className="text-body font-semibold text-white">
                      {item.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section
        className="section animate-fade-in-up"
        style={{ animationDelay: "0.6s" }}
      >
        <div className="bg-amber-500/5 border border-amber-300/20 rounded-2xl p-8">
          <div className="flex items-start space-x-4">
            <div className="p-3 bg-amber-400/20 rounded-xl border border-amber-200/30">
              <AlertTriangle className="w-6 h-6 text-amber-200" />
            </div>
            <div>
              <h4 className="text-title text-amber-50 mb-2">
                Important Disclaimer
              </h4>
              <p className="text-body text-amber-100/90 leading-relaxed">
                This system is designed as a decision-support tool for medical
                professionals and should not replace professional medical
                judgment. Always consult with qualified healthcare providers for
                diagnosis and treatment decisions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Contact */}
      <section
        className="section animate-fade-in-up"
        style={{ animationDelay: "0.65s" }}
      >
        <div className="bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl text-center">
          <h2 className="text-headline text-white mb-4">Get in Touch</h2>
          <p className="text-body text-slate-300 mb-8 max-w-lg mx-auto">
            Have questions or feedback about DeepBreast AI? We'd love to hear
            from you.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary"
            >
              <Github className="w-5 h-5" />
              <span>GitHub</span>
            </a>
            <a href="mailto:contact@example.com" className="btn-primary">
              <Mail className="w-5 h-5" />
              <span>Contact Us</span>
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <section
        className="section animate-fade-in-up"
        style={{ animationDelay: "0.7s" }}
      >
        <div className="text-center text-body text-slate-400">
          <p>DeepBreast AI v2.0.0 • Built with React, FastAPI & PyTorch</p>
          <p className="mt-2">© 2024 All Rights Reserved</p>
        </div>
      </section>
    </div>
  );
};

export default About;
