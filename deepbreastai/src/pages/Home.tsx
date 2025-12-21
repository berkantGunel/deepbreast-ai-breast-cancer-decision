import { Brain, Upload, BarChart3, Eye, ArrowRight, Zap, Shield, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className="page-container">
      {/* Hero Section */}
      <section className="section">
        <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-800/50 via-slate-900/50 to-slate-950/50 border border-white/10 p-8 md:p-12 lg:p-16">
          {/* Background Decorations */}
          <div className="absolute top-0 right-0 w-80 h-80 bg-emerald-500/20 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/4" />
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-cyan-500/20 rounded-full blur-[100px] translate-y-1/2 -translate-x-1/4" />
          <div className="absolute top-1/3 right-1/4 w-40 h-40 bg-purple-500/15 rounded-full blur-[80px]" />

          <div className="relative z-10 flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
            {/* Text Content */}
            <div className="flex-1 max-w-2xl">
              {/* Status Badge */}
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-full mb-6">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-sm font-semibold text-emerald-400">
                  Live • 92.86% Test Accuracy
                </span>
              </div>

              {/* Heading */}
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white leading-tight mb-6">
                Precision AI for Early{" "}
                <span className="text-gradient">Breast Cancer Detection</span>
              </h1>

              {/* Description */}
              <p className="text-lg text-slate-300 leading-relaxed mb-8">
                ResNet18 transfer learning, 277K+ histopathology slides, CUDA-accelerated
                inference & clinician-friendly explanations in one seamless platform.
              </p>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 mb-8">
                <Link
                  to="/predict"
                  className="btn-primary group"
                >
                  <Brain className="w-5 h-5" />
                  <span>Start Analysis</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Link>
                <Link
                  to="/analysis"
                  className="btn-secondary"
                >
                  <Eye className="w-5 h-5" />
                  <span>View Explainability</span>
                </Link>
              </div>

              {/* Feature Tags */}
              <div className="flex flex-wrap gap-3">
                <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-sm text-slate-300">
                  <Shield className="w-4 h-4 text-emerald-400" />
                  HIPAA-aware workflow
                </span>
                <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-sm text-slate-300">
                  <Zap className="w-4 h-4 text-amber-400" />
                  {"< 2s"} GPU inference
                </span>
              </div>
            </div>

            {/* Hero Illustration */}
            <div className="hidden lg:block flex-shrink-0">
              <div className="relative w-72 h-72 xl:w-80 xl:h-80">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/30 to-cyan-500/30 rounded-3xl blur-2xl animate-pulse" style={{ animationDuration: "3s" }} />
                <div className="relative w-full h-full bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-white/20 rounded-3xl flex items-center justify-center shadow-2xl">
                  <Brain
                    className="w-28 h-28 xl:w-32 xl:h-32 text-white"
                    style={{ filter: "drop-shadow(0 0 30px rgba(16, 185, 129, 0.5))" }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="section">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
          {[
            { label: "Accuracy", value: "95.4%", color: "text-emerald-400", bg: "from-emerald-500/10 to-green-500/5" },
            { label: "Training Images", value: "277K+", color: "text-cyan-400", bg: "from-cyan-500/10 to-blue-500/5" },
            { label: "Inference Time", value: "<2s", color: "text-purple-400", bg: "from-purple-500/10 to-indigo-500/5" },
            { label: "Classes", value: "2", color: "text-amber-400", bg: "from-amber-500/10 to-orange-500/5" },
          ].map((stat) => (
            <div
              key={stat.label}
              className={`glass-card bg-gradient-to-br ${stat.bg} p-6 text-center hover:scale-105 transition-transform duration-300`}
            >
              <div className={`text-3xl lg:text-4xl font-bold ${stat.color} mb-2`}>
                {stat.value}
              </div>
              <div className="text-sm lg:text-base text-slate-300 font-medium">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section className="section">
        <div className="text-center mb-12">
          <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
            Key Features
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Built for clinical workflows with transparency, speed, and rigor.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {[
            {
              title: "Image Upload & Analysis",
              desc: "Upload histopathology images and get instant AI predictions with confidence scores. Supports PNG, JPG and TIFF formats.",
              icon: Upload,
              iconBg: "from-emerald-500/20 to-cyan-500/20",
              iconColor: "text-emerald-400",
            },
            {
              title: "Explainable AI (Grad-CAM)",
              desc: "Visualize which regions of the image influenced the model's decision with intuitive heatmap overlays.",
              icon: Eye,
              iconBg: "from-purple-500/20 to-indigo-500/20",
              iconColor: "text-purple-400",
            },
            {
              title: "Performance Metrics",
              desc: "View detailed model metrics including confusion matrix, training curves, and validation statistics.",
              icon: BarChart3,
              iconBg: "from-amber-500/20 to-orange-500/20",
              iconColor: "text-amber-400",
            },
            {
              title: "Real-time Processing",
              desc: "GPU-accelerated inference delivers results in under 2 seconds for rapid clinical workflows.",
              icon: Zap,
              iconBg: "from-pink-500/20 to-rose-500/20",
              iconColor: "text-pink-400",
            },
          ].map((feature) => (
            <div
              key={feature.title}
              className="glass-card p-6 lg:p-8 group"
            >
              <div className="flex items-start gap-5">
                <div className={`flex-shrink-0 p-4 rounded-2xl bg-gradient-to-br ${feature.iconBg} border border-white/10 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className={`w-6 h-6 ${feature.iconColor}`} />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-slate-400 leading-relaxed">
                    {feature.desc}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works Section */}
      <section className="section">
        <div className="text-center mb-12">
          <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
            How It Works
          </h2>
          <p className="text-lg text-slate-400">
            Simple 3-step process for instant analysis
          </p>
        </div>

        <div className="glass-card-static p-8 lg:p-12">
          <div className="grid md:grid-cols-3 gap-8 lg:gap-12">
            {[
              {
                step: "1",
                title: "Upload Image",
                desc: "Select a histopathology image from your device",
                gradient: "from-emerald-500 to-cyan-500",
              },
              {
                step: "2",
                title: "AI Analysis",
                desc: "Our CNN model processes and analyzes the tissue",
                gradient: "from-cyan-500 to-blue-500",
              },
              {
                step: "3",
                title: "Get Results",
                desc: "View diagnosis, confidence scores, and heatmaps",
                gradient: "from-blue-500 to-purple-500",
              },
            ].map((item, index) => (
              <div key={item.title} className="text-center relative">
                {/* Connector line for desktop */}
                {index < 2 && (
                  <div className="hidden md:block absolute top-10 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-white/20 to-transparent" />
                )}

                <div className={`relative w-16 h-16 lg:w-20 lg:h-20 bg-gradient-to-br ${item.gradient} rounded-2xl flex items-center justify-center text-white text-2xl lg:text-3xl font-bold mx-auto mb-6 shadow-xl`}>
                  {item.step}
                </div>
                <h4 className="text-lg lg:text-xl font-semibold text-white mb-3">
                  {item.title}
                </h4>
                <p className="text-slate-400">
                  {item.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Medical Disclaimer */}
      <section>
        <div className="glass-card-static bg-gradient-to-br from-amber-500/5 to-orange-500/5 border-amber-500/20 p-6 lg:p-8">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl">
              <Shield className="w-6 h-6 text-amber-400" />
            </div>
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-amber-200 mb-2">
                Medical Disclaimer
              </h4>
              <p className="text-amber-100/80 leading-relaxed">
                This AI system is intended as a decision-support tool for medical professionals only.
                It should not be used as a sole diagnostic method. Always consult with qualified
                healthcare providers for medical decisions.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;