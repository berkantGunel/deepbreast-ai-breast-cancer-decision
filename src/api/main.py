"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import predict, gradcam, metrics, report, dicom, history, mammography

# Initialize FastAPI app
app = FastAPI(
    title="DeepBreast API",
    description="AI-Based Breast Cancer Detection API with Uncertainty Estimation, PDF Reports, DICOM Support, and Analysis History",
    version="2.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allows frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(gradcam.router, prefix="/api", tags=["Grad-CAM"])
app.include_router(metrics.router, prefix="/api", tags=["Metrics"])
app.include_router(report.router, prefix="/api", tags=["Reports"])
app.include_router(dicom.router, prefix="/api", tags=["DICOM"])
app.include_router(history.router, prefix="/api", tags=["History & Batch"])
app.include_router(mammography.router, prefix="/api", tags=["Mammography"])

@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "DeepBreast API is running",
        "version": "2.3.0",
        "features": [
            "Cancer Detection (Histopathology)",
            "Mammography Analysis (BI-RADS)",
            "MC Dropout Uncertainty",
            "Grad-CAM Visualization",
            "PDF Reports",
            "DICOM Support",
            "Analysis History",
            "Batch Upload"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


