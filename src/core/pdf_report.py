"""
PDF Report Generator Module

Generates professional medical analysis reports in PDF format.
Includes:
- Patient/Case information
- Image analysis results
- Uncertainty metrics
- Grad-CAM visualization
- Clinical recommendations
- Medical disclaimer
"""

import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image as RLImage, PageBreak, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.piecharts import Pie


class PDFReportGenerator:
    """
    Generates professional PDF reports for breast cancer analysis results.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.page_width, self.page_height = A4
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0f172a')
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#64748b')
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#1e293b'),
            borderPadding=5
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#334155'),
            leading=14
        ))
        
        # Result text (large)
        self.styles.add(ParagraphStyle(
            name='ResultText',
            parent=self.styles['Normal'],
            fontSize=28,
            spaceBefore=10,
            spaceAfter=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#10b981')
        ))
        
        # Disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=TA_JUSTIFY,
            leading=10
        ))
        
        # Metric label
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#64748b')
        ))
        
        # Metric value
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#1e293b'),
            fontName='Helvetica-Bold'
        ))

    def _create_header(self, case_id: str, timestamp: str) -> list:
        """Create report header with logo and title."""
        elements = []
        
        # Title
        elements.append(Paragraph("🩺 DeepBreast AI", self.styles['ReportTitle']))
        elements.append(Paragraph(
            "Breast Cancer Detection Analysis Report",
            self.styles['ReportSubtitle']
        ))
        
        # Horizontal line
        elements.append(HRFlowable(
            width="100%", thickness=1, 
            color=colors.HexColor('#e2e8f0'),
            spaceBefore=10, spaceAfter=20
        ))
        
        # Report info table
        report_info = [
            ['Case ID:', case_id],
            ['Report Date:', timestamp],
            ['Model Version:', 'ResNet18 v2.0'],
            ['Analysis Type:', 'Histopathology Classification']
        ]
        
        info_table = Table(report_info, colWidths=[2.5*cm, 8*cm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1e293b')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_diagnosis_section(self, prediction: str, confidence: float) -> list:
        """Create the main diagnosis section."""
        elements = []
        
        elements.append(Paragraph("Diagnosis Result", self.styles['SectionHeader']))
        
        # Diagnosis box
        is_malignant = prediction.lower() == 'malignant'
        bg_color = colors.HexColor('#fef2f2') if is_malignant else colors.HexColor('#ecfdf5')
        text_color = colors.HexColor('#dc2626') if is_malignant else colors.HexColor('#10b981')
        border_color = colors.HexColor('#fecaca') if is_malignant else colors.HexColor('#a7f3d0')
        
        # Create styled result
        result_style = ParagraphStyle(
            name='DiagnosisResult',
            parent=self.styles['Normal'],
            fontSize=32,
            alignment=TA_CENTER,
            textColor=text_color,
            fontName='Helvetica-Bold'
        )
        
        # Result table (acts as a box)
        diagnosis_data = [
            [Paragraph(prediction.upper(), result_style)],
            [Paragraph(
                f"Confidence: {confidence:.1f}%",
                ParagraphStyle(
                    name='ConfidenceText',
                    parent=self.styles['Normal'],
                    fontSize=14,
                    alignment=TA_CENTER,
                    textColor=text_color
                )
            )]
        ]
        
        diagnosis_table = Table(diagnosis_data, colWidths=[14*cm])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 2, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(diagnosis_table)
        elements.append(Spacer(1, 15))
        
        # Description
        if is_malignant:
            desc = "The AI analysis indicates the presence of <b>malignant (cancerous)</b> tissue patterns in the sample. Further clinical evaluation is strongly recommended."
        else:
            desc = "The AI analysis indicates <b>benign (non-cancerous)</b> tissue patterns in the sample. However, clinical confirmation is still recommended."
        
        elements.append(Paragraph(desc, self.styles['BodyText']))
        
        return elements

    def _create_probability_section(self, probabilities: Dict[str, float]) -> list:
        """Create probability distribution section."""
        elements = []
        
        elements.append(Paragraph("Probability Distribution", self.styles['SectionHeader']))
        
        # Probability table
        prob_data = [
            ['Class', 'Probability', 'Visual'],
        ]
        
        for class_name, prob in probabilities.items():
            # Create visual bar
            bar_width = int(prob * 2)  # Scale to max 200
            color = '#10b981' if class_name.lower() == 'benign' else '#ef4444'
            bar = f'{"█" * max(1, bar_width // 10)}' + f' {prob:.1f}%'
            
            prob_data.append([
                class_name.capitalize(),
                f"{prob:.1f}%",
                bar
            ])
        
        prob_table = Table(prob_data, colWidths=[3*cm, 3*cm, 8*cm])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#475569')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#10b981')),  # Benign
            ('TEXTCOLOR', (1, 2), (1, 2), colors.HexColor('#ef4444')),  # Malignant
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(prob_table)
        
        return elements

    def _create_uncertainty_section(self, uncertainty: Dict[str, Any], reliability: str, recommendation: str) -> list:
        """Create uncertainty metrics section."""
        elements = []
        
        elements.append(Paragraph("Uncertainty Analysis (MC Dropout)", self.styles['SectionHeader']))
        
        # Reliability badge color
        rel_colors = {
            'high': ('#10b981', '#ecfdf5', 'High Reliability'),
            'medium': ('#f59e0b', '#fffbeb', 'Medium Reliability'),
            'low': ('#ef4444', '#fef2f2', 'Low Reliability')
        }
        
        rel_text_color, rel_bg_color, rel_label = rel_colors.get(
            reliability.lower(), 
            ('#64748b', '#f1f5f9', 'Unknown')
        )
        
        # Reliability box
        rel_style = ParagraphStyle(
            name='ReliabilityLabel',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor(rel_text_color),
            fontName='Helvetica-Bold'
        )
        
        rel_data = [[Paragraph(f"⬤ {rel_label}", rel_style)]]
        rel_table = Table(rel_data, colWidths=[14*cm])
        rel_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(rel_bg_color)),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(rel_text_color)),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(rel_table)
        elements.append(Spacer(1, 10))
        
        # Uncertainty metrics
        metrics_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Uncertainty Score', f"{uncertainty.get('score', 0):.1f}%", 'Overall prediction uncertainty (lower is better)'],
            ['Entropy', f"{uncertainty.get('entropy', 0):.4f}", 'Information-theoretic uncertainty'],
            ['Epistemic', f"{uncertainty.get('epistemic', 0):.4f}", 'Model knowledge uncertainty'],
            ['Coef. of Variation', f"{uncertainty.get('coefficient_of_variation', 0):.1f}%", 'Prediction consistency'],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[4*cm, 3*cm, 7*cm])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#475569')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (2, 1), (2, -1), colors.HexColor('#64748b')),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 10))
        
        # Clinical recommendation
        elements.append(Paragraph("<b>Clinical Recommendation:</b>", self.styles['BodyText']))
        elements.append(Paragraph(recommendation, self.styles['BodyText']))
        
        return elements

    def _create_image_section(self, original_image_b64: Optional[str], gradcam_image_b64: Optional[str]) -> list:
        """Create image analysis section with original and Grad-CAM."""
        elements = []
        
        elements.append(Paragraph("Image Analysis", self.styles['SectionHeader']))
        
        images = []
        labels = []
        
        if original_image_b64:
            try:
                img_data = base64.b64decode(original_image_b64)
                img = Image.open(io.BytesIO(img_data))
                
                # Save to temp buffer for ReportLab
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                rl_img = RLImage(img_buffer, width=6*cm, height=6*cm)
                images.append(rl_img)
                labels.append('Original Image')
            except Exception as e:
                elements.append(Paragraph(f"[Original image unavailable: {str(e)}]", self.styles['BodyText']))
        
        if gradcam_image_b64:
            try:
                img_data = base64.b64decode(gradcam_image_b64)
                img = Image.open(io.BytesIO(img_data))
                
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                rl_img = RLImage(img_buffer, width=6*cm, height=6*cm)
                images.append(rl_img)
                labels.append('Grad-CAM Heatmap')
            except Exception as e:
                elements.append(Paragraph(f"[Grad-CAM unavailable: {str(e)}]", self.styles['BodyText']))
        
        if images:
            # Create image table
            img_data = [images, [Paragraph(l, self.styles['MetricLabel']) for l in labels]]
            img_table = Table(img_data, colWidths=[7*cm] * len(images))
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(img_table)
        else:
            elements.append(Paragraph("[No images available]", self.styles['BodyText']))
        
        return elements

    def _create_disclaimer(self) -> list:
        """Create medical disclaimer section."""
        elements = []
        
        elements.append(Spacer(1, 30))
        elements.append(HRFlowable(
            width="100%", thickness=1,
            color=colors.HexColor('#e2e8f0'),
            spaceBefore=10, spaceAfter=10
        ))
        
        disclaimer_text = """
        <b>MEDICAL DISCLAIMER:</b> This report is generated by an artificial intelligence system 
        and is intended for use as a decision-support tool by qualified medical professionals only. 
        This AI analysis should NOT be used as the sole basis for diagnosis or treatment decisions. 
        The predictions and uncertainty estimates provided are based on mathematical models and may 
        not account for all clinical factors. Any medical decisions should be made in consultation 
        with qualified healthcare providers who have access to the complete clinical picture, 
        including patient history, physical examination, and additional diagnostic tests. 
        The developers of this system accept no liability for clinical decisions made based on 
        this report. Always seek professional medical advice for health concerns.
        """
        
        elements.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))
        
        # Footer
        elements.append(Spacer(1, 15))
        footer_text = f"Generated by DeepBreast AI v2.0 • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • For Research Purposes Only"
        elements.append(Paragraph(footer_text, ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#94a3b8')
        )))
        
        return elements

    def generate_report(
        self,
        prediction_result: Dict[str, Any],
        original_image_b64: Optional[str] = None,
        gradcam_image_b64: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> bytes:
        """
        Generate a complete PDF report.
        
        Args:
            prediction_result: Prediction API response with uncertainty metrics
            original_image_b64: Base64 encoded original image
            gradcam_image_b64: Base64 encoded Grad-CAM heatmap
            case_id: Optional case identifier
            
        Returns:
            PDF file as bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        elements = []
        
        # Generate case ID if not provided
        if not case_id:
            case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        timestamp = datetime.now().strftime('%B %d, %Y at %H:%M')
        
        # Add sections
        elements.extend(self._create_header(case_id, timestamp))
        
        # Diagnosis section
        elements.extend(self._create_diagnosis_section(
            prediction_result.get('prediction', 'Unknown'),
            prediction_result.get('confidence', 0)
        ))
        
        elements.append(Spacer(1, 10))
        
        # Probability section
        probabilities = prediction_result.get('probabilities', {})
        if probabilities:
            elements.extend(self._create_probability_section(probabilities))
        
        # Uncertainty section (if MC Dropout enabled)
        if prediction_result.get('mc_dropout_enabled') and prediction_result.get('uncertainty'):
            elements.extend(self._create_uncertainty_section(
                prediction_result['uncertainty'],
                prediction_result.get('reliability', 'unknown'),
                prediction_result.get('clinical_recommendation', 'No recommendation available.')
            ))
        
        # Image section
        if original_image_b64 or gradcam_image_b64:
            elements.extend(self._create_image_section(original_image_b64, gradcam_image_b64))
        
        # Disclaimer
        elements.extend(self._create_disclaimer())
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes


def generate_analysis_report(
    prediction_result: Dict[str, Any],
    original_image_b64: Optional[str] = None,
    gradcam_image_b64: Optional[str] = None,
    case_id: Optional[str] = None
) -> bytes:
    """
    Convenience function to generate PDF report.
    
    Args:
        prediction_result: Prediction API response
        original_image_b64: Base64 encoded original image
        gradcam_image_b64: Base64 encoded Grad-CAM heatmap
        case_id: Optional case identifier
        
    Returns:
        PDF file as bytes
    """
    generator = PDFReportGenerator()
    return generator.generate_report(
        prediction_result,
        original_image_b64,
        gradcam_image_b64,
        case_id
    )
