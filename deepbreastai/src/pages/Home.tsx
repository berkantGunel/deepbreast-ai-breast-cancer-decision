import { Link } from "react-router-dom";
import {
  Brain,
  Upload,
  BarChart3,
  Eye,
  ArrowRight,
  Zap,
  Shield,
  Target,
} from "lucide-react";

const Home = () => {
  return (
    <div>
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-100">
        <h1 className="text-3xl font-bold text-gray-900">
          Welcome to DeepBreast AI
        </h1>
        <p className="text-gray-600 mt-1">
          AI-powered breast cancer detection from histopathology images
        </p>
      </div>

      <div className="p-8 max-w-6xl">
        {/* Hero Card */}
        <div className="bg-gradient-to-r from-red-500 to-pink-600 rounded-2xl p-8 mb-8 text-white shadow-xl">
          <div className="flex items-start justify-between">
            <div className="max-w-2xl">
              <div className="inline-flex items-center px-3 py-1 bg-white/20 backdrop-blur rounded-full text-sm font-medium mb-4">
                🏆 95.4% Accuracy on Test Set
              </div>
              <h2 className="text-4xl font-bold mb-4">
                Deep Learning for Early Cancer Detection
              </h2>
              <p className="text-lg text-white/90 mb-6 leading-relaxed">
                Our advanced CNN model analyzes histopathology images to detect
                malignant tissue with clinical-grade accuracy. Trained on
                277,000+ medical images.
              </p>
              <Link
                to="/predict"
                className="inline-flex items-center space-x-2 bg-white text-red-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-all shadow-lg hover:shadow-xl"
              >
                <Brain className="w-5 h-5" />
                <span>Start Analysis</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
            <div className="hidden lg:block">
              <div className="w-32 h-32 bg-white/20 backdrop-blur rounded-2xl flex items-center justify-center">
                <Brain className="w-16 h-16 text-white" />
              </div>
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-50 rounded-xl p-5 border border-gray-100">
            <p className="text-3xl font-bold text-gray-900">95.4%</p>
            <p className="text-sm text-gray-600 mt-1">Accuracy</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-5 border border-gray-100">
            <p className="text-3xl font-bold text-gray-900">277K+</p>
            <p className="text-sm text-gray-600 mt-1">Training Images</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-5 border border-gray-100">
            <p className="text-3xl font-bold text-gray-900">&lt;2s</p>
            <p className="text-sm text-gray-600 mt-1">Inference Time</p>
          </div>
          <div className="bg-gray-50 rounded-xl p-5 border border-gray-100">
            <p className="text-3xl font-bold text-gray-900">2</p>
            <p className="text-sm text-gray-600 mt-1">Classes</p>
          </div>
        </div>

        {/* Section Title */}
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          Key Features
        </h3>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md transition-shadow">
            <div className="flex items-start space-x-4">
              <div className="p-3 bg-red-50 rounded-lg">
                <Upload className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 mb-1">
                  Image Upload & Analysis
                </h4>
                <p className="text-sm text-gray-600">
                  Upload histopathology images and get instant AI predictions
                  with confidence scores
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md transition-shadow">
            <div className="flex items-start space-x-4">
              <div className="p-3 bg-blue-50 rounded-lg">
                <Eye className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 mb-1">
                  Explainable AI (Grad-CAM)
                </h4>
                <p className="text-sm text-gray-600">
                  Visualize which regions of the image influenced the model's
                  decision
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md transition-shadow">
            <div className="flex items-start space-x-4">
              <div className="p-3 bg-green-50 rounded-lg">
                <BarChart3 className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 mb-1">
                  Performance Metrics
                </h4>
                <p className="text-sm text-gray-600">
                  View detailed model metrics including confusion matrix and
                  training history
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-md transition-shadow">
            <div className="flex items-start space-x-4">
              <div className="p-3 bg-purple-50 rounded-lg">
                <Zap className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 mb-1">
                  Real-time Processing
                </h4>
                <p className="text-sm text-gray-600">
                  GPU-accelerated inference delivers results in under 2 seconds
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* How It Works */}
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          How It Works
        </h3>
        <div className="bg-gray-50 rounded-xl p-6 border border-gray-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center text-red-600 font-bold">
                1
              </div>
              <div>
                <p className="font-medium text-gray-900">Upload Image</p>
                <p className="text-sm text-gray-500">
                  Select histopathology image
                </p>
              </div>
            </div>
            <ArrowRight className="w-5 h-5 text-gray-400" />
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center text-red-600 font-bold">
                2
              </div>
              <div>
                <p className="font-medium text-gray-900">AI Analysis</p>
                <p className="text-sm text-gray-500">CNN processes the image</p>
              </div>
            </div>
            <ArrowRight className="w-5 h-5 text-gray-400" />
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center text-red-600 font-bold">
                3
              </div>
              <div>
                <p className="font-medium text-gray-900">Get Results</p>
                <p className="text-sm text-gray-500">
                  View diagnosis & heatmap
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <Shield className="w-5 h-5 text-yellow-600 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-yellow-800">
                Medical Disclaimer
              </p>
              <p className="text-sm text-yellow-700">
                This tool is for educational and research purposes only. Always
                consult healthcare professionals for medical decisions.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
