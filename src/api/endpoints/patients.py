"""Patient management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from src.api.database import get_db
from src.api.models.user import User
from src.api.models.patient import Patient, Analysis
from src.api.utils.auth import get_current_user_required

router = APIRouter()


# Pydantic models
class PatientCreate(BaseModel):
    patient_id: Optional[str] = None
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    medical_history: Optional[str] = None
    notes: Optional[str] = None


class PatientUpdate(BaseModel):
    patient_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    medical_history: Optional[str] = None
    notes: Optional[str] = None


class AnalysisResponse(BaseModel):
    id: int
    analysis_type: str
    prediction: str
    confidence: float
    birads_category: Optional[str] = None
    clinical_recommendation: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class PatientResponse(BaseModel):
    id: int
    patient_id: Optional[str]
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime]
    gender: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    medical_history: Optional[str]
    notes: Optional[str]
    created_at: datetime
    analysis_count: int = 0
    
    class Config:
        from_attributes = True


class PatientDetailResponse(PatientResponse):
    analyses: List[AnalysisResponse] = []
    address: Optional[str] = None


class AnalysisCreate(BaseModel):
    analysis_type: str  # histopathology, mammography
    prediction: str
    predicted_class: Optional[int] = None
    confidence: float
    probabilities: Optional[dict] = None
    birads_category: Optional[str] = None
    uncertainty_score: Optional[float] = None
    reliability: Optional[str] = None
    clinical_recommendation: Optional[str] = None
    mc_dropout_enabled: Optional[bool] = False
    n_samples: Optional[int] = None
    image_filename: Optional[str] = None
    notes: Optional[str] = None


class AnalysisDetailResponse(AnalysisResponse):
    patient_id: int
    predicted_class: Optional[int]
    probabilities: Optional[dict]
    uncertainty_score: Optional[float]
    reliability: Optional[str]
    mc_dropout_enabled: int
    n_samples: Optional[int]
    image_filename: Optional[str]
    notes: Optional[str]
    doctor_notes: Optional[str]
    updated_at: datetime


# Patient endpoints
@router.get("/patients", response_model=List[PatientResponse])
async def list_patients(
    search: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """List all patients for current user."""
    query = db.query(Patient).filter(Patient.user_id == current_user.id)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Patient.first_name.ilike(search_term)) |
            (Patient.last_name.ilike(search_term)) |
            (Patient.patient_id.ilike(search_term))
        )
    
    patients = query.order_by(desc(Patient.updated_at)).offset(skip).limit(limit).all()
    
    # Add analysis count
    result = []
    for patient in patients:
        patient_data = PatientResponse.model_validate(patient)
        patient_data.analysis_count = len(patient.analyses)
        result.append(patient_data)
    
    return result


@router.post("/patients", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient_data: PatientCreate,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Create a new patient."""
    patient = Patient(
        user_id=current_user.id,
        patient_id=patient_data.patient_id,
        first_name=patient_data.first_name,
        last_name=patient_data.last_name,
        date_of_birth=patient_data.date_of_birth,
        gender=patient_data.gender,
        phone=patient_data.phone,
        email=patient_data.email,
        address=patient_data.address,
        medical_history=patient_data.medical_history,
        notes=patient_data.notes,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    
    return patient


@router.get("/patients/{patient_id}", response_model=PatientDetailResponse)
async def get_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get patient details with analyses."""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    response = PatientDetailResponse.model_validate(patient)
    response.analysis_count = len(patient.analyses)
    response.analyses = [AnalysisResponse.model_validate(a) for a in patient.analyses]
    
    return response


@router.put("/patients/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    patient_data: PatientUpdate,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Update patient information."""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    update_data = patient_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)
    
    patient.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(patient)
    
    return patient


@router.delete("/patients/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Delete a patient and all their analyses."""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    db.delete(patient)
    db.commit()
    return None


# Analysis endpoints
@router.post("/patients/{patient_id}/analyses", response_model=AnalysisDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    patient_id: int,
    analysis_data: AnalysisCreate,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Add an analysis to a patient."""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    analysis = Analysis(
        patient_id=patient.id,
        analysis_type=analysis_data.analysis_type,
        prediction=analysis_data.prediction,
        predicted_class=analysis_data.predicted_class,
        confidence=analysis_data.confidence,
        probabilities=analysis_data.probabilities,
        birads_category=analysis_data.birads_category,
        uncertainty_score=analysis_data.uncertainty_score,
        reliability=analysis_data.reliability,
        clinical_recommendation=analysis_data.clinical_recommendation,
        mc_dropout_enabled=1 if analysis_data.mc_dropout_enabled else 0,
        n_samples=analysis_data.n_samples,
        image_filename=analysis_data.image_filename,
        notes=analysis_data.notes,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    return analysis


@router.get("/patients/{patient_id}/analyses", response_model=List[AnalysisDetailResponse])
async def list_patient_analyses(
    patient_id: int,
    analysis_type: Optional[str] = None,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """List all analyses for a patient."""
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    query = db.query(Analysis).filter(Analysis.patient_id == patient_id)
    
    if analysis_type:
        query = query.filter(Analysis.analysis_type == analysis_type)
    
    analyses = query.order_by(desc(Analysis.created_at)).all()
    return analyses


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetailResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get analysis details."""
    analysis = db.query(Analysis).join(Patient).filter(
        Analysis.id == analysis_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    return analysis


@router.put("/analyses/{analysis_id}/notes")
async def update_analysis_notes(
    analysis_id: int,
    notes: Optional[str] = None,
    doctor_notes: Optional[str] = None,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Update analysis notes."""
    analysis = db.query(Analysis).join(Patient).filter(
        Analysis.id == analysis_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    if notes is not None:
        analysis.notes = notes
    if doctor_notes is not None:
        analysis.doctor_notes = doctor_notes
    
    analysis.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Notes updated successfully"}


@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Delete an analysis."""
    analysis = db.query(Analysis).join(Patient).filter(
        Analysis.id == analysis_id,
        Patient.user_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    db.delete(analysis)
    db.commit()
    return None


# Statistics
@router.get("/patients/stats/summary")
async def get_patient_stats(
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get patient and analysis statistics."""
    patient_count = db.query(Patient).filter(Patient.user_id == current_user.id).count()
    
    analysis_count = db.query(Analysis).join(Patient).filter(
        Patient.user_id == current_user.id
    ).count()
    
    # Analysis type breakdown
    histopathology_count = db.query(Analysis).join(Patient).filter(
        Patient.user_id == current_user.id,
        Analysis.analysis_type == "histopathology"
    ).count()
    
    mammography_count = db.query(Analysis).join(Patient).filter(
        Patient.user_id == current_user.id,
        Analysis.analysis_type == "mammography"
    ).count()
    
    # Prediction breakdown
    malignant_count = db.query(Analysis).join(Patient).filter(
        Patient.user_id == current_user.id,
        Analysis.prediction == "Malignant"
    ).count()
    
    benign_count = db.query(Analysis).join(Patient).filter(
        Patient.user_id == current_user.id,
        Analysis.prediction == "Benign"
    ).count()
    
    return {
        "total_patients": patient_count,
        "total_analyses": analysis_count,
        "by_type": {
            "histopathology": histopathology_count,
            "mammography": mammography_count
        },
        "by_prediction": {
            "benign": benign_count,
            "malignant": malignant_count
        }
    }
