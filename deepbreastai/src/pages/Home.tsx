import { Brain, Upload, BarChart3, Eye, ArrowRight, Zap, Shield } from "lucide-react";

const Home = () => {
  return (
    <div className="w-full">
      <div className="w-full px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto">
        {/* Hero Section */}
        <section className="mb-16 lg:mb-24">
          <div className="relative overflow-hidden rounded-3xl p-8 sm:p-10 lg:p-16 shadow-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950 border border-white/5">
            {/* Background decorations */}
            <div className="absolute top-0 right-0 w-72 h-72 lg:w-96 lg:h-96 bg-purple-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-0 pointer-events-none" />
            <div className="absolute bottom-0 left-0 w-48 h-48 lg:w-64 lg:h-64 bg-cyan-400/10 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2 pointer-events-none" />
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(255,255,255,0.05),transparent_40%)] pointer-events-none" />

            <div className="relative z-10 flex flex-col lg:flex-row items-center justify-between gap-8 lg:gap-12">
              <div className="max-w-2xl space-y-5 lg:space-y-6">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-md rounded-full text-sm font-semibold shadow-lg">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Live • 92.86% test accuracy
                </div>
                
                <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black leading-tight text-white">
                  Precision AI for Early Breast Cancer Detection
                </h1>
                
                <p className="text-base sm:text-lg lg:text-xl text-slate-200/90 leading-relaxed">
                  ResNet18 transfer learning, 277K+ slides, CUDA-accelerated inference & clinician-friendly explanations in one place.
                </p>
                
                <div className="flex flex-col sm:flex-row gap-4 pt-2">
                  <a
                    href="/predict"
                    className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-400 to-cyan-400 text-slate-950 font-bold shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105"
                  >
                    <Brain className="w-5 h-5" />
                    Start Analysis
                    <ArrowRight className="w-4 h-4" />
                  </a>
                  <a
                    href="/analysis"
                    className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl border border-white/20 text-white font-semibold hover:bg-white/10 transition-all duration-300"
                  >
                    <Eye className="w-5 h-5" />
                    View Explainability
                  </a>
                </div>
                
                <div className="flex flex-wrap gap-3 pt-2">
                  <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-sm text-slate-200/90">
                    <Shield className="w-4 h-4" /> HIPAA-aware workflow
                  </span>
                  <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-sm text-slate-200/90">
                    <Zap className="w-4 h-4" /> &lt; 2s GPU inference
                  </span>
                </div>
              </div>
              
              <div className="hidden lg:block flex-shrink-0">
                <div className="relative w-64 h-64 xl:w-72 xl:h-72 bg-white/5 border border-white/10 rounded-3xl flex items-center justify-center shadow-2xl backdrop-blur-xl">
                  <div className="absolute inset-8 rounded-3xl bg-gradient-to-br from-pink-500/30 to-indigo-500/30 blur-2xl animate-pulse" />
                  <Brain className="relative w-20 h-20 xl:w-24 xl:h-24 text-white drop-shadow-2xl" />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Stats Row */}
        <section className="mb-16 lg:mb-24">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
            {[
              { label: "Accuracy", value: "95.4%" },
              { label: "Training Images", value: "277K+" },
              { label: "Inference Time", value: "<2s" },
              { label: "Classes", value: "2" },
            ].map((item) => (
              <div
                key={item.label}
                className="bg-white/5 border border-white/10 rounded-2xl p-6 text-center shadow-lg hover:shadow-xl transition-shadow duration-300"
              >
                <div className="text-3xl lg:text-4xl font-bold text-white mb-2">
                  {item.value}
                </div>
                <div className="text-sm lg:text-base text-slate-300">
                  {item.label}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Features Section */}
        <section className="mb-16 lg:mb-24">
          <div className="text-center mb-10 lg:mb-12">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
              Key Features
            </h2>
            <p className="text-base lg:text-lg text-slate-300 max-w-2xl mx-auto">
              Built for clinical workflows with transparency, speed, and rigor.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 gap-6 lg:gap-8">
            {[
              {
                title: "Image Upload & Analysis",
                desc: "Upload histopathology images and get instant AI predictions with confidence scores. Supports PNG, JPG and TIFF formats.",
                icon: <Upload className="w-7 h-7" />,
                accent: "from-emerald-400/20 to-cyan-400/10",
              },
              {
                title: "Explainable AI (Grad-CAM)",
                desc: "Visualize which regions of the image influenced the model's decision with intuitive heatmap overlays.",
                icon: <Eye className="w-7 h-7" />,
                accent: "from-indigo-400/20 to-purple-400/10",
              },
              {
                title: "Performance Metrics",
                desc: "View detailed model metrics including confusion matrix, training curves, and validation statistics.",
                icon: <BarChart3 className="w-7 h-7" />,
                accent: "from-amber-400/20 to-orange-400/10",
              },
              {
                title: "Real-time Processing",
                desc: "GPU-accelerated inference delivers results in under 2 seconds for rapid clinical workflows.",
                icon: <Zap className="w-7 h-7" />,
                accent: "from-pink-400/20 to-red-400/10",
              },
            ].map((feature) => (
              <div
                key={feature.title}
                className="bg-gradient-to-br from-slate-900 to-slate-950 border border-white/5 rounded-2xl p-6 lg:p-8 hover:shadow-2xl hover:border-white/10 transition-all duration-300 group"
              >
                <div className="flex items-start gap-4 lg:gap-5">
                  <div className={`flex-shrink-0 p-3 lg:p-4 rounded-xl bg-gradient-to-br ${feature.accent} border border-white/10 text-white group-hover:scale-110 transition-transform duration-300`}> 
                    {feature.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg lg:text-xl font-bold text-white mb-2 lg:mb-3">
                      {feature.title}
                    </h3>
                    <p className="text-sm lg:text-base text-slate-300 leading-relaxed">
                      {feature.desc}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* How It Works */}
        <section className="mb-16 lg:mb-24">
          <div className="text-center mb-10 lg:mb-12">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4">
              How It Works
            </h2>
            <p className="text-base lg:text-lg text-slate-300">
              Simple 3-step process for instant analysis
            </p>
          </div>

          <div className="bg-white/5 border border-white/10 rounded-2xl p-8 lg:p-12 shadow-xl">
            <div className="grid md:grid-cols-3 gap-8 lg:gap-12">
              {[
                {
                  step: "1",
                  title: "Upload Image",
                  desc: "Select a histopathology image from your device",
                },
                {
                  step: "2",
                  title: "AI Analysis",
                  desc: "Our CNN model processes and analyzes the tissue",
                },
                {
                  step: "3",
                  title: "Get Results",
                  desc: "View diagnosis, confidence scores, and heatmaps",
                },
              ].map((item) => (
                <div className="text-center" key={item.title}>
                  <div className="w-14 h-14 lg:w-16 lg:h-16 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center text-white text-xl lg:text-2xl font-bold mx-auto mb-4 lg:mb-5 shadow-lg">
                    {item.step}
                  </div>
                  <h4 className="text-lg lg:text-xl font-bold text-white mb-2 lg:mb-3">
                    {item.title}
                  </h4>
                  <p className="text-sm lg:text-base text-slate-300">
                    {item.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <section>
          <div className="bg-amber-500/5 border border-amber-300/20 rounded-2xl p-6 lg:p-8">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 p-3 bg-amber-400/20 rounded-xl border border-amber-200/30">
                <Shield className="w-5 h-5 lg:w-6 lg:h-6 text-amber-200" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-lg lg:text-xl font-bold text-amber-50 mb-2">
                  Medical Disclaimer
                </h4>
                <p className="text-sm lg:text-base text-amber-100/90 leading-relaxed">
                  This AI system is intended as a decision-support tool for
                  medical professionals only. It should not be used as a sole
                  diagnostic method. Always consult with qualified healthcare
                  providers for medical decisions.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Home;