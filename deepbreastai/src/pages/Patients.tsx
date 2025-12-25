import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
    Users,
    Plus,
    Search,
    Calendar,
    Phone,
    Mail,
    Activity,
    ChevronRight,
    Edit2,
    Trash2,
    X,
    Loader2,
    User,
    FileText,
    AlertCircle,
} from 'lucide-react';
import './Patients.css';

interface Patient {
    id: number;
    patient_id: string | null;
    first_name: string;
    last_name: string;
    date_of_birth: string | null;
    gender: string | null;
    phone: string | null;
    email: string | null;
    medical_history: string | null;
    notes: string | null;
    created_at: string;
    analysis_count: number;
}

interface PatientStats {
    total_patients: number;
    total_analyses: number;
    by_type: {
        histopathology: number;
        mammography: number;
    };
    by_prediction: {
        benign: number;
        malignant: number;
    };
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Patients = () => {
    const navigate = useNavigate();
    const { token, isAuthenticated, isLoading: authLoading } = useAuth();

    const [patients, setPatients] = useState<Patient[]>([]);
    const [stats, setStats] = useState<PatientStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [showModal, setShowModal] = useState(false);
    const [editingPatient, setEditingPatient] = useState<Patient | null>(null);
    const [formData, setFormData] = useState({
        patient_id: '',
        first_name: '',
        last_name: '',
        date_of_birth: '',
        gender: '',
        phone: '',
        email: '',
        medical_history: '',
        notes: '',
    });
    const [formError, setFormError] = useState('');
    const [submitting, setSubmitting] = useState(false);

    useEffect(() => {
        if (!authLoading && !isAuthenticated) {
            navigate('/auth');
            return;
        }
        if (token) {
            fetchPatients();
            fetchStats();
        }
    }, [token, isAuthenticated, authLoading, navigate]);

    const fetchPatients = async (search?: string) => {
        try {
            const url = search
                ? `${API_BASE_URL}/api/patients?search=${encodeURIComponent(search)}`
                : `${API_BASE_URL}/api/patients`;

            const response = await fetch(url, {
                headers: { 'Authorization': `Bearer ${token}` },
            });

            if (response.ok) {
                const data = await response.json();
                setPatients(data);
            }
        } catch (error) {
            console.error('Failed to fetch patients:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchStats = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/patients/stats/summary`, {
                headers: { 'Authorization': `Bearer ${token}` },
            });

            if (response.ok) {
                const data = await response.json();
                setStats(data);
            }
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    };

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        fetchPatients(searchTerm);
    };

    const openCreateModal = () => {
        setEditingPatient(null);
        setFormData({
            patient_id: '',
            first_name: '',
            last_name: '',
            date_of_birth: '',
            gender: '',
            phone: '',
            email: '',
            medical_history: '',
            notes: '',
        });
        setFormError('');
        setShowModal(true);
    };

    const openEditModal = (patient: Patient) => {
        setEditingPatient(patient);
        setFormData({
            patient_id: patient.patient_id || '',
            first_name: patient.first_name,
            last_name: patient.last_name,
            date_of_birth: patient.date_of_birth?.split('T')[0] || '',
            gender: patient.gender || '',
            phone: patient.phone || '',
            email: patient.email || '',
            medical_history: patient.medical_history || '',
            notes: patient.notes || '',
        });
        setFormError('');
        setShowModal(true);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setFormError('');
        setSubmitting(true);

        try {
            const url = editingPatient
                ? `${API_BASE_URL}/api/patients/${editingPatient.id}`
                : `${API_BASE_URL}/api/patients`;

            const method = editingPatient ? 'PUT' : 'POST';

            const response = await fetch(url, {
                method,
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...formData,
                    date_of_birth: formData.date_of_birth || null,
                }),
            });

            if (response.ok) {
                setShowModal(false);
                fetchPatients();
                fetchStats();
            } else {
                const error = await response.json();
                setFormError(error.detail || 'Failed to save patient');
            }
        } catch {
            setFormError('An error occurred');
        } finally {
            setSubmitting(false);
        }
    };

    const handleDelete = async (patient: Patient) => {
        if (!confirm(`Are you sure you want to delete ${patient.first_name} ${patient.last_name}? This will also delete all their analyses.`)) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/patients/${patient.id}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` },
            });

            if (response.ok) {
                fetchPatients();
                fetchStats();
            }
        } catch (error) {
            console.error('Failed to delete patient:', error);
        }
    };

    if (authLoading || loading) {
        return (
            <div className="patients-page">
                <div className="loading-container">
                    <Loader2 className="spinner" size={40} />
                    <p>Loading patients...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="patients-page">
            {/* Header */}
            <div className="patients-header">
                <div>
                    <h1>
                        <Users size={32} />
                        Patient Management
                    </h1>
                    <p>Manage patients and their analysis history</p>
                </div>
                <button className="add-patient-btn" onClick={openCreateModal}>
                    <Plus size={20} />
                    Add Patient
                </button>
            </div>

            {/* Stats Cards */}
            {stats && (
                <div className="stats-grid">
                    <div className="stat-card patients">
                        <div className="stat-icon">
                            <Users size={24} />
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{stats.total_patients}</span>
                            <span className="stat-label">Total Patients</span>
                        </div>
                    </div>
                    <div className="stat-card analyses">
                        <div className="stat-icon">
                            <FileText size={24} />
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{stats.total_analyses}</span>
                            <span className="stat-label">Total Analyses</span>
                        </div>
                    </div>
                    <div className="stat-card benign">
                        <div className="stat-icon">
                            <Activity size={24} />
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{stats.by_prediction.benign}</span>
                            <span className="stat-label">Benign Results</span>
                        </div>
                    </div>
                    <div className="stat-card malignant">
                        <div className="stat-icon">
                            <AlertCircle size={24} />
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{stats.by_prediction.malignant}</span>
                            <span className="stat-label">Malignant Results</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Search */}
            <form onSubmit={handleSearch} className="search-form">
                <div className="search-input-wrapper">
                    <Search size={20} />
                    <input
                        type="text"
                        placeholder="Search patients by name or ID..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
                <button type="submit" className="search-btn">Search</button>
            </form>

            {/* Patients List */}
            <div className="patients-list">
                {patients.length === 0 ? (
                    <div className="empty-state">
                        <User size={48} />
                        <h3>No Patients Found</h3>
                        <p>Add your first patient to get started</p>
                        <button onClick={openCreateModal} className="add-first-btn">
                            <Plus size={18} />
                            Add Patient
                        </button>
                    </div>
                ) : (
                    patients.map((patient) => (
                        <div key={patient.id} className="patient-card">
                            <div className="patient-avatar">
                                {patient.first_name[0]}{patient.last_name[0]}
                            </div>
                            <div className="patient-info">
                                <h3>{patient.first_name} {patient.last_name}</h3>
                                <div className="patient-meta">
                                    {patient.patient_id && (
                                        <span className="meta-item">
                                            <FileText size={14} />
                                            ID: {patient.patient_id}
                                        </span>
                                    )}
                                    {patient.date_of_birth && (
                                        <span className="meta-item">
                                            <Calendar size={14} />
                                            {new Date(patient.date_of_birth).toLocaleDateString()}
                                        </span>
                                    )}
                                    {patient.phone && (
                                        <span className="meta-item">
                                            <Phone size={14} />
                                            {patient.phone}
                                        </span>
                                    )}
                                    {patient.email && (
                                        <span className="meta-item">
                                            <Mail size={14} />
                                            {patient.email}
                                        </span>
                                    )}
                                </div>
                            </div>
                            <div className="patient-stats">
                                <div className="analysis-count">
                                    <span className="count">{patient.analysis_count}</span>
                                    <span className="label">Analyses</span>
                                </div>
                            </div>
                            <div className="patient-actions">
                                <button
                                    className="action-btn view"
                                    onClick={() => navigate(`/patients/${patient.id}`)}
                                    title="View Details"
                                >
                                    <ChevronRight size={20} />
                                </button>
                                <button
                                    className="action-btn edit"
                                    onClick={() => openEditModal(patient)}
                                    title="Edit"
                                >
                                    <Edit2 size={18} />
                                </button>
                                <button
                                    className="action-btn delete"
                                    onClick={() => handleDelete(patient)}
                                    title="Delete"
                                >
                                    <Trash2 size={18} />
                                </button>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Add/Edit Modal */}
            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>{editingPatient ? 'Edit Patient' : 'Add New Patient'}</h2>
                            <button className="close-btn" onClick={() => setShowModal(false)}>
                                <X size={20} />
                            </button>
                        </div>

                        {formError && (
                            <div className="modal-error">
                                <AlertCircle size={18} />
                                {formError}
                            </div>
                        )}

                        <form onSubmit={handleSubmit} className="patient-form">
                            <div className="form-row">
                                <div className="form-group">
                                    <label>First Name *</label>
                                    <input
                                        type="text"
                                        value={formData.first_name}
                                        onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label>Last Name *</label>
                                    <input
                                        type="text"
                                        value={formData.last_name}
                                        onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                                        required
                                    />
                                </div>
                            </div>

                            <div className="form-row">
                                <div className="form-group">
                                    <label>Patient ID</label>
                                    <input
                                        type="text"
                                        value={formData.patient_id}
                                        onChange={(e) => setFormData({ ...formData, patient_id: e.target.value })}
                                        placeholder="Hospital ID"
                                    />
                                </div>
                                <div className="form-group">
                                    <label>Date of Birth</label>
                                    <input
                                        type="date"
                                        value={formData.date_of_birth}
                                        onChange={(e) => setFormData({ ...formData, date_of_birth: e.target.value })}
                                    />
                                </div>
                            </div>

                            <div className="form-row">
                                <div className="form-group">
                                    <label>Gender</label>
                                    <select
                                        value={formData.gender}
                                        onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                                    >
                                        <option value="">Select...</option>
                                        <option value="Female">Female</option>
                                        <option value="Male">Male</option>
                                        <option value="Other">Other</option>
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label>Phone</label>
                                    <input
                                        type="tel"
                                        value={formData.phone}
                                        onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                                    />
                                </div>
                            </div>

                            <div className="form-group">
                                <label>Email</label>
                                <input
                                    type="email"
                                    value={formData.email}
                                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                />
                            </div>

                            <div className="form-group">
                                <label>Medical History</label>
                                <textarea
                                    value={formData.medical_history}
                                    onChange={(e) => setFormData({ ...formData, medical_history: e.target.value })}
                                    rows={3}
                                    placeholder="Previous conditions, family history..."
                                />
                            </div>

                            <div className="form-group">
                                <label>Notes</label>
                                <textarea
                                    value={formData.notes}
                                    onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                                    rows={2}
                                    placeholder="Additional notes..."
                                />
                            </div>

                            <div className="form-actions">
                                <button type="button" className="cancel-btn" onClick={() => setShowModal(false)}>
                                    Cancel
                                </button>
                                <button type="submit" className="submit-btn" disabled={submitting}>
                                    {submitting ? (
                                        <>
                                            <Loader2 size={18} className="spinner" />
                                            Saving...
                                        </>
                                    ) : (
                                        editingPatient ? 'Update Patient' : 'Add Patient'
                                    )}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Patients;
