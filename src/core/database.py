"""
Database Module - SQLite for Analysis History

Provides:
- Connection management
- Analysis history storage
- Batch analysis support
- Query operations
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64
import io
from contextlib import contextmanager


# Database file location
DB_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DB_DIR / "deepbreast.db"


def ensure_db_dir():
    """Ensure database directory exists."""
    DB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection():
    """Get database connection with context manager."""
    ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize database tables."""
    ensure_db_dir()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Analysis history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                file_type TEXT DEFAULT 'image',
                
                -- Prediction results
                prediction TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                confidence REAL NOT NULL,
                prob_benign REAL NOT NULL,
                prob_malignant REAL NOT NULL,
                
                -- Uncertainty metrics (MC Dropout)
                mc_dropout_enabled INTEGER DEFAULT 0,
                uncertainty_score REAL,
                uncertainty_entropy REAL,
                uncertainty_epistemic REAL,
                reliability TEXT,
                clinical_recommendation TEXT,
                
                -- Optional stored data
                thumbnail TEXT,  -- Base64 encoded small preview
                
                -- Metadata
                notes TEXT,
                tags TEXT,  -- JSON array of tags
                is_batch INTEGER DEFAULT 0,
                batch_id TEXT
            )
        """)
        
        # Batch analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_analysis (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_files INTEGER NOT NULL,
                completed_files INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                summary TEXT  -- JSON summary
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_created 
            ON analysis_history(created_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_prediction 
            ON analysis_history(prediction)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_batch 
            ON analysis_history(batch_id)
        """)
        
        conn.commit()
        
    print(f"✅ Database initialized at: {DB_PATH}")


class AnalysisHistoryDB:
    """Database operations for analysis history."""
    
    def __init__(self):
        init_database()
    
    def save_analysis(
        self,
        filename: str,
        prediction_result: Dict[str, Any],
        thumbnail_b64: Optional[str] = None,
        file_type: str = "image",
        notes: str = "",
        tags: List[str] = None,
        is_batch: bool = False,
        batch_id: str = None
    ) -> int:
        """
        Save analysis result to database.
        
        Args:
            filename: Original filename
            prediction_result: API prediction response
            thumbnail_b64: Optional base64 thumbnail
            file_type: 'image' or 'dicom'
            notes: Optional notes
            tags: Optional list of tags
            is_batch: Part of batch analysis
            batch_id: Batch identifier
            
        Returns:
            Record ID
        """
        uncertainty = prediction_result.get('uncertainty', {})
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analysis_history (
                    filename, file_type, prediction, predicted_class,
                    confidence, prob_benign, prob_malignant,
                    mc_dropout_enabled, uncertainty_score, uncertainty_entropy,
                    uncertainty_epistemic, reliability, clinical_recommendation,
                    thumbnail, notes, tags, is_batch, batch_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                file_type,
                prediction_result.get('prediction', prediction_result.get('class_name', 'Unknown')),
                prediction_result.get('predicted_class', prediction_result.get('prediction', 0)),
                prediction_result.get('confidence', 0),
                prediction_result.get('probabilities', {}).get('benign', prediction_result.get('mean_probs', [0, 0])[0]),
                prediction_result.get('probabilities', {}).get('malignant', prediction_result.get('mean_probs', [0, 0])[1]),
                1 if prediction_result.get('mc_dropout_enabled') else 0,
                uncertainty.get('score'),
                uncertainty.get('entropy'),
                uncertainty.get('epistemic'),
                prediction_result.get('reliability'),
                prediction_result.get('clinical_recommendation'),
                thumbnail_b64,
                notes,
                json.dumps(tags) if tags else None,
                1 if is_batch else 0,
                batch_id
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        prediction_filter: str = None,
        search: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get analysis history.
        
        Args:
            limit: Max records to return
            offset: Pagination offset
            prediction_filter: Filter by 'Benign' or 'Malignant'
            search: Search in filename or notes
            
        Returns:
            List of history records
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM analysis_history WHERE 1=1"
            params = []
            
            if prediction_filter:
                query += " AND prediction = ?"
                params.append(prediction_filter)
            
            if search:
                query += " AND (filename LIKE ? OR notes LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]
    
    def get_analysis(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Get single analysis by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None
    
    def delete_analysis(self, record_id: int) -> bool:
        """Delete analysis record."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM analysis_history")
            total = cursor.fetchone()[0]
            
            # By prediction
            cursor.execute("""
                SELECT prediction, COUNT(*) as count 
                FROM analysis_history 
                GROUP BY prediction
            """)
            by_prediction = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By reliability
            cursor.execute("""
                SELECT reliability, COUNT(*) as count 
                FROM analysis_history 
                WHERE reliability IS NOT NULL
                GROUP BY reliability
            """)
            by_reliability = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM analysis_history")
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Recent analyses (last 7 days)
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM analysis_history
                WHERE created_at >= datetime('now', '-7 days')
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """)
            daily_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_analyses': total,
                'by_prediction': by_prediction,
                'by_reliability': by_reliability,
                'average_confidence': round(avg_confidence, 2),
                'daily_counts': daily_counts
            }
    
    def clear_history(self) -> int:
        """Clear all history. Returns number of deleted records."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analysis_history")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM analysis_history")
            conn.commit()
            return count
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite Row to dictionary."""
        if row is None:
            return None
        
        d = dict(row)
        
        # Parse JSON fields
        if d.get('tags'):
            try:
                d['tags'] = json.loads(d['tags'])
            except:
                d['tags'] = []
        
        # Convert boolean fields
        d['mc_dropout_enabled'] = bool(d.get('mc_dropout_enabled'))
        d['is_batch'] = bool(d.get('is_batch'))
        
        return d


# Batch analysis management
class BatchAnalysisDB:
    """Database operations for batch analysis."""
    
    def __init__(self):
        init_database()
    
    def create_batch(self, batch_id: str, total_files: int) -> str:
        """Create new batch record."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO batch_analysis (id, total_files, status)
                VALUES (?, ?, 'processing')
            """, (batch_id, total_files))
            conn.commit()
        return batch_id
    
    def update_batch_progress(self, batch_id: str, completed: int):
        """Update batch progress."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE batch_analysis 
                SET completed_files = ?
                WHERE id = ?
            """, (completed, batch_id))
            conn.commit()
    
    def complete_batch(self, batch_id: str, summary: Dict[str, Any]):
        """Mark batch as complete with summary."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE batch_analysis 
                SET status = 'completed', 
                    summary = ?,
                    completed_files = total_files
                WHERE id = ?
            """, (json.dumps(summary), batch_id))
            conn.commit()
    
    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch info."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM batch_analysis WHERE id = ?", (batch_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get('summary'):
                    d['summary'] = json.loads(d['summary'])
                return d
            return None
    
    def get_batch_analyses(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all analyses for a batch."""
        history_db = AnalysisHistoryDB()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analysis_history 
                WHERE batch_id = ?
                ORDER BY id
            """, (batch_id,))
            rows = cursor.fetchall()
            return [history_db._row_to_dict(row) for row in rows]


# Convenience functions
def get_history_db() -> AnalysisHistoryDB:
    """Get history database instance."""
    return AnalysisHistoryDB()


def get_batch_db() -> BatchAnalysisDB:
    """Get batch database instance."""
    return BatchAnalysisDB()
