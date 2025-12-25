"""Patient and Analysis models."""

from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from src.api.database import Base


class Patient(Base):
    """Patient model for grouping analyses."""
    
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Patient Information
    patient_id = Column(String(100), index=True)  # Hospital/External ID
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(20), nullable=True)
    phone = Column(String(50), nullable=True)
    email = Column(String(255), nullable=True)
    address = Column(Text, nullable=True)
    
    # Medical Information
    medical_history = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="patients")
    analyses = relationship("Analysis", back_populates="patient", cascade="all, delete-orphan")
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    @property
    def analysis_count(self):
        return len(self.analyses)
    
    def __repr__(self):
        return f"<Patient(id={self.id}, name='{self.full_name}')>"


class Analysis(Base):
    """Analysis model for storing prediction results."""
    
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    
    # Analysis Type
    analysis_type = Column(String(50), nullable=False)  # histopathology, mammography
    
    # Image Information
    image_filename = Column(String(255), nullable=True)
    image_path = Column(String(500), nullable=True)
    
    # Prediction Results
    prediction = Column(String(100), nullable=False)
    predicted_class = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=True)
    
    # Additional Results
    birads_category = Column(String(50), nullable=True)  # For mammography
    uncertainty_score = Column(Float, nullable=True)
    reliability = Column(String(50), nullable=True)
    clinical_recommendation = Column(Text, nullable=True)
    
    # MC Dropout
    mc_dropout_enabled = Column(Integer, default=0)  # Boolean as int for SQLite
    n_samples = Column(Integer, nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    doctor_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="analyses")
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, type='{self.analysis_type}', prediction='{self.prediction}')>"
