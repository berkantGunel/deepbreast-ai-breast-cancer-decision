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
} from "lucide-react";

const About = () => {
  const features = [
    {
      icon: Brain,
      title: "Deep Learning",
      description:
        "Advanced CNN architecture trained on 277,000+ histopathology images for exceptional accuracy in cancer detection.",
      color: "text-blue-600",
      bg: "bg-blue-50",
    },
    {
      icon: Target,
      title: "95.4% Accuracy",
      description:
        "Validated performance metrics demonstrating clinical-grade accuracy in distinguishing benign from malignant tissue.",
      color: "text-green-600",
      bg: "bg-green-50",
    },
    {
      icon: Zap,
      title: "Real-time Analysis",
      description:
        "Lightning-fast inference with results in under 2 seconds, enabling rapid clinical decision-making.",
      color: "text-yellow-600",
      bg: "bg-yellow-50",
    },
    {
      icon: Shield,
      title: "Explainable AI",
      description:
        "Grad-CAM visualizations show exactly which tissue regions influenced the AI's diagnosis.",
      color: "text-purple-600",
      bg: "bg-purple-50",
    },
    {
      icon: Users,
      title: "Clinical Support",
      description:
        "Designed as a decision support tool for pathologists, augmenting human expertise with AI capabilities.",
      color: "text-pink-600",
      bg: "bg-pink-50",
    },
    {
      icon: Award,
      title: "Research-Grade",
      description:
        "Built on peer-reviewed methodologies and validated using established medical imaging datasets.",
      color: "text-indigo-600",
      bg: "bg-indigo-50",
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
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">
          About DeepBreast AI
        </h1>
        <p className="text-gray-600">
          AI-Powered Breast Cancer Detection System
        </p>
      </div>

      {/* Project Info Card */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-white/20 rounded-lg">
            <GraduationCap className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-xl font-semibold mb-2">Graduation Project</h2>
            <p className="text-white/90 text-sm leading-relaxed">
              This project demonstrates the application of deep learning in
              medical imaging for breast cancer detection. Combining
              state-of-the-art convolutional neural networks with explainable AI
              techniques to provide accurate, transparent, and reliable
              diagnostic support.
            </p>
            <div className="flex gap-2 mt-4">
              <span className="px-3 py-1 bg-white/20 rounded-full text-xs font-medium">
                Deep Learning
              </span>
              <span className="px-3 py-1 bg-white/20 rounded-full text-xs font-medium">
                Medical AI
              </span>
              <span className="px-3 py-1 bg-white/20 rounded-full text-xs font-medium">
                Computer Vision
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mission */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Our Mission
        </h2>
        <p className="text-gray-600 text-sm leading-relaxed mb-3">
          DeepBreast AI leverages cutting-edge deep learning technology to
          assist medical professionals in the early detection of breast cancer
          through histopathology image analysis.
        </p>
        <p className="text-gray-600 text-sm leading-relaxed">
          <strong className="text-gray-800">
            Early detection saves lives.
          </strong>{" "}
          Our goal is to make AI-powered diagnostic tools accessible and
          trustworthy, empowering healthcare providers with advanced technology
          while maintaining the highest standards of accuracy and
          interpretability.
        </p>
      </div>

      {/* Features Grid */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Key Features
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature, index) => (
            <div key={index} className="card hover:shadow-md transition-shadow">
              <div className={`p-2.5 ${feature.bg} rounded-lg w-fit mb-3`}>
                <feature.icon className={`w-5 h-5 ${feature.color}`} />
              </div>
              <h3 className="font-medium text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Technical Specifications */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-gray-600" />
          Technical Specifications
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Model Architecture
            </h3>
            <div className="space-y-2">
              {techSpecs.architecture.map((item, index) => (
                <div
                  key={index}
                  className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0"
                >
                  <span className="text-sm text-gray-600">{item.label}</span>
                  <span className="text-sm font-medium text-gray-900">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
              <Database className="w-4 h-4" />
              Training Details
            </h3>
            <div className="space-y-2">
              {techSpecs.training.map((item, index) => (
                <div
                  key={index}
                  className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0"
                >
                  <span className="text-sm text-gray-600">{item.label}</span>
                  <span className="text-sm font-medium text-gray-900">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Contact */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Get in Touch
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          For questions, collaborations, or feedback about this project:
        </p>
        <div className="flex gap-3">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors"
          >
            <Github className="w-4 h-4" />
            View on GitHub
          </a>
          <a
            href="mailto:contact@deepbreastai.com"
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Mail className="w-4 h-4" />
            Contact
          </a>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <div className="flex gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-amber-800 mb-1">
              Medical Disclaimer
            </h3>
            <p className="text-sm text-amber-700">
              This system is designed for research and educational purposes
              only. It should not be used as a substitute for professional
              medical diagnosis or treatment. Always consult qualified
              healthcare professionals for medical decisions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
