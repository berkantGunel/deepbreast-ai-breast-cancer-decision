"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.endpoints import predict, gradcam, metrics, report, dicom, history, auth, patients, mammography_classical, mammography, segmentation
from src.api.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup: Initialize database
    init_db()
    print("✅ Database initialized")
    yield
    # Shutdown: cleanup if needed
    print("👋 Application shutting down")


# Initialize FastAPI app
app = FastAPI(
    title="DeepBreast API",
    description="AI-Based Breast Cancer Detection API with Uncertainty Estimation, PDF Reports, DICOM Support, User Authentication, and Patient Management",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(patients.router, prefix="/api", tags=["Patients & Analyses"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(gradcam.router, prefix="/api", tags=["Grad-CAM"])
app.include_router(metrics.router, prefix="/api", tags=["Metrics"])
app.include_router(report.router, prefix="/api", tags=["Reports"])
app.include_router(dicom.router, prefix="/api", tags=["DICOM"])
app.include_router(history.router, prefix="/api", tags=["History & Batch"])
app.include_router(mammography_classical.router, prefix="/api", tags=["Mammography Classical"])
app.include_router(mammography.router, prefix="/api", tags=["Mammography"])
app.include_router(segmentation.router, tags=["Tumor Segmentation"])

@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "DeepBreast API is running",
        "version": "3.1.0",
        "features": [
            "User Authentication (JWT)",
            "Patient Management",
            "Cancer Detection (Histopathology)",
            "Mammography Analysis (Classical ML - DMID)",
            "MC Dropout Uncertainty",
            "Grad-CAM Visualization",
            "PDF Reports",
            "DICOM Support",
            "Analysis History",
            "Batch Upload",
            "Tumor Segmentation (U-Net)"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/health")
async def api_health():
    """API Health check endpoint for Dashboard."""
    return {"status": "healthy"}
