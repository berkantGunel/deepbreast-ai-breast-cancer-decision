"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import predict, gradcam, metrics

# Initialize FastAPI app
app = FastAPI(
    title="DeepBreast API",
    description="AI-Based Breast Cancer Detection API",
    version="2.0.0",
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

@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "DeepBreast API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
