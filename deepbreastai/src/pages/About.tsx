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
      description: "Advanced CNN architecture trained on 277,000+ histopathology images for exceptional accuracy.",
      gradient: "from-cyan-500/20 to-blue-500/20",
      iconColor: "text-cyan-400",
    },
    {
      icon: Target,
      title: "95.4% Accuracy",
      description: "Validated performance demonstrating clinical-grade accuracy in cancer detection.",
      gradient: "from-emerald-500/20 to-green-500/20",
      iconColor: "text-emerald-400",
    },
    {
      icon: Zap,
      title: "Real-time Analysis",
      description: "Lightning-fast inference with results in under 2 seconds for rapid decisions.",
      gradient: "from-amber-500/20 to-orange-500/20",
      iconColor: "text-amber-400",
    },
    {
      icon: Shield,
      title: "Explainable AI",
      description: "Grad-CAM visualizations show which tissue regions influenced the diagnosis.",
      gradient: "from-purple-500/20 to-violet-500/20",
      iconColor: "text-purple-400",
    },
    {
      icon: Users,
      title: "Clinical Support",
      description: "Decision support tool for pathologists, augmenting human expertise with AI.",
      gradient: "from-pink-500/20 to-rose-500/20",
      iconColor: "text-pink-400",
    },
    {
      icon: Award,
      title: "Research-Grade",
      description: "Built on peer-reviewed methodologies and validated medical imaging datasets.",
      gradient: "from-indigo-500/20 to-blue-500/20",
      iconColor: "text-indigo-400",
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
    <div className="page-container">
      {/* Page Header */}
      <section className="section">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30 rounded-2xl">
            <Heart className="w-8 h-8 text-purple-400" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-bold text-white">
              About DeepBreast AI
            </h1>
            <p className="text-slate-400 mt-1">
              AI-Powered Breast Cancer Detection System
            </p>
          </div>
        </div>
      </section>

      {/* Project Info Card */}
      <section className="section">
        <div className="glass-card relative overflow-hidden p-8 lg:p-10">
          {/* Background decoration */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/10 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/4" />

          <div className="relative z-10 flex flex-col lg:flex-row items-start gap-6">
            <div className="p-4 bg-white/10 border border-white/20 rounded-2xl">
              <GraduationCap className="w-10 h-10 text-emerald-400" />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-white mb-4">
                Graduation Project
              </h2>
              <p className="text-slate-300 leading-relaxed mb-6">
                This project demonstrates the application of deep learning in medical imaging
                for breast cancer detection. Combining state-of-the-art convolutional neural
                networks with explainable AI techniques to provide accurate, transparent, and
                reliable diagnostic support.
              </p>
              <div className="flex flex-wrap gap-2">
                {["Deep Learning", "Medical AI", "Computer Vision", "Explainable AI"].map((tag) => (
                  <span
                    key={tag}
                    className="px-4 py-2 bg-white/10 border border-white/10 rounded-lg text-sm font-medium text-slate-200"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Mission */}
      <section className="section">
        <div className="glass-card p-8">
          <h2 className="text-2xl font-bold text-white mb-4">Our Mission</h2>
          <div className="space-y-4 text-slate-300 leading-relaxed">
            <p>
              DeepBreast AI leverages cutting-edge deep learning technology to assist medical
              professionals in the early detection of breast cancer through histopathology image analysis.
            </p>
            <p>
              <strong className="text-white">Early detection saves lives.</strong> Our goal is to
              make AI-powered diagnostic tools accessible and trustworthy, empowering healthcare
              providers with advanced technology while maintaining the highest standards of accuracy
              and interpretability.
            </p>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="section">
        <h2 className="text-2xl font-bold text-white mb-6">Key Features</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((feature, index) => (
            <div
              key={index}
              className="glass-card p-6 group"
            >
              <div className={`p-4 bg-gradient-to-br ${feature.gradient} border border-white/10 rounded-2xl w-fit mb-5 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className={`w-6 h-6 ${feature.iconColor}`} />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Technical Specifications */}
      <section className="section">
        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2 bg-cyan-500/20 border border-cyan-500/30 rounded-lg">
              <Cpu className="w-5 h-5 text-cyan-400" />
            </div>
            <h2 className="text-2xl font-bold text-white">Technical Specifications</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-10">
            {/* Architecture */}
            <div>
              <div className="flex items-center gap-3 mb-5">
                <Brain className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">Model Architecture</h3>
              </div>
              <div className="space-y-3">
                {techSpecs.architecture.map((item, index) => (
                  <div
                    key={index}
                    className="flex justify-between items-center py-3 border-b border-white/10 last:border-0"
                  >
                    <span className="text-slate-400">{item.label}</span>
                    <span className="text-white font-medium text-right">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Training */}
            <div>
              <div className="flex items-center gap-3 mb-5">
                <Database className="w-5 h-5 text-emerald-400" />
                <h3 className="text-lg font-semibold text-white">Training Details</h3>
              </div>
              <div className="space-y-3">
                {techSpecs.training.map((item, index) => (
                  <div
                    key={index}
                    className="flex justify-between items-center py-3 border-b border-white/10 last:border-0"
                  >
                    <span className="text-slate-400">{item.label}</span>
                    <span className="text-white font-medium text-right">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="section">
        <div className="glass-card bg-gradient-to-br from-amber-500/5 to-orange-500/5 border-amber-500/20 p-6">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h4 className="font-semibold text-amber-200 mb-2">Important Disclaimer</h4>
              <p className="text-amber-100/80 text-sm leading-relaxed">
                This system is designed as a decision-support tool for medical professionals and
                should not replace professional medical judgment. Always consult with qualified
                healthcare providers for diagnosis and treatment decisions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Contact */}
      <section className="section">
        <div className="glass-card p-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-3">Get in Touch</h2>
          <p className="text-slate-400 mb-6 max-w-lg mx-auto">
            Have questions or feedback about DeepBreast AI? We'd love to hear from you.
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
      <section className="text-center text-slate-500 text-sm">
        <p>DeepBreast AI v2.0.0 • Built with React, FastAPI & PyTorch</p>
        <p className="mt-1">© 2024 All Rights Reserved</p>
      </section>
    </div>
  );
};

export default About;
