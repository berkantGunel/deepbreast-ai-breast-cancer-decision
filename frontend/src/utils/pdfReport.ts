/**
 * PDF Report Generator Utility
 * Creates professional PDF reports for analysis results using browser's print functionality
 */

export interface ReportData {
    type: "histopathology" | "mammography";
    prediction: string;
    confidence: number;
    probabilities: Record<string, number>;
    biradsCategory?: string;
    recommendation?: {
        action: string;
        urgency: string;
        description: string;
        next_steps: string[];
    };
    imageUrl?: string;
    timestamp: string;
    patientId?: string;
    notes?: string;
}

const formatDate = (date: Date): string => {
    return date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
};

const getUrgencyColor = (urgency: string): string => {
    switch (urgency) {
        case "low": return "#22c55e";
        case "medium": return "#f59e0b";
        case "high": return "#ef4444";
        default: return "#6b7280";
    }
};

const getPredictionColor = (prediction: string): string => {
    const pred = prediction.toLowerCase();
    if (pred.includes("benign")) return "#22c55e";
    if (pred.includes("malignant")) return "#ef4444";
    if (pred.includes("suspicious")) return "#f59e0b";
    return "#6b7280";
};

export const generatePDFReport = async (data: ReportData): Promise<void> => {
    const reportId = `RPT-${Date.now().toString(36).toUpperCase()}`;

    // Create a new window for the report
    const printWindow = window.open("", "_blank");
    if (!printWindow) {
        throw new Error("Could not open print window. Please allow popups.");
    }

    const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DeepBreast AI Report - ${reportId}</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.6;
      padding: 20px;
    }
    
    .report-container {
      max-width: 800px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
      overflow: hidden;
    }
    
    .header {
      background: linear-gradient(135deg, #8b5cf6, #06b6d4);
      color: white;
      padding: 30px;
      text-align: center;
    }
    
    .header h1 {
      font-size: 28px;
      margin-bottom: 5px;
    }
    
    .header p {
      opacity: 0.9;
      font-size: 14px;
    }
    
    .report-id {
      margin-top: 15px;
      font-size: 12px;
      opacity: 0.8;
    }
    
    .content {
      padding: 30px;
    }
    
    .section {
      margin-bottom: 25px;
      padding-bottom: 20px;
      border-bottom: 1px solid #e2e8f0;
    }
    
    .section:last-child {
      border-bottom: none;
      margin-bottom: 0;
    }
    
    .section-title {
      font-size: 16px;
      font-weight: 600;
      color: #64748b;
      margin-bottom: 15px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    
    .result-card {
      background: ${getPredictionColor(data.prediction)}15;
      border-left: 4px solid ${getPredictionColor(data.prediction)};
      padding: 20px;
      border-radius: 0 12px 12px 0;
    }
    
    .result-prediction {
      font-size: 32px;
      font-weight: 700;
      color: ${getPredictionColor(data.prediction)};
      margin-bottom: 10px;
    }
    
    .result-confidence {
      font-size: 18px;
      color: #475569;
    }
    
    .birads-badge {
      display: inline-block;
      background: ${getPredictionColor(data.prediction)};
      color: white;
      padding: 5px 15px;
      border-radius: 20px;
      font-weight: 600;
      margin-top: 10px;
    }
    
    .probability-grid {
      display: grid;
      grid-template-columns: repeat(${Object.keys(data.probabilities).length}, 1fr);
      gap: 15px;
    }
    
    .probability-item {
      text-align: center;
      padding: 15px;
      background: #f1f5f9;
      border-radius: 12px;
    }
    
    .probability-label {
      font-size: 12px;
      color: #64748b;
      margin-bottom: 5px;
    }
    
    .probability-value {
      font-size: 24px;
      font-weight: 700;
      color: #1e293b;
    }
    
    .recommendation-card {
      background: ${data.recommendation ? getUrgencyColor(data.recommendation.urgency) + "15" : "#f1f5f9"};
      border-left: 4px solid ${data.recommendation ? getUrgencyColor(data.recommendation.urgency) : "#64748b"};
      padding: 20px;
      border-radius: 0 12px 12px 0;
    }
    
    .recommendation-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 15px;
    }
    
    .recommendation-action {
      font-size: 18px;
      font-weight: 600;
      color: #1e293b;
    }
    
    .urgency-badge {
      background: ${data.recommendation ? getUrgencyColor(data.recommendation.urgency) : "#64748b"};
      color: white;
      padding: 5px 12px;
      border-radius: 15px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    
    .recommendation-description {
      color: #475569;
      margin-bottom: 15px;
    }
    
    .next-steps {
      list-style: none;
      padding: 0;
    }
    
    .next-steps li {
      padding: 8px 0;
      padding-left: 25px;
      position: relative;
      color: #334155;
    }
    
    .next-steps li::before {
      content: "→";
      position: absolute;
      left: 0;
      color: ${data.recommendation ? getUrgencyColor(data.recommendation.urgency) : "#64748b"};
      font-weight: bold;
    }
    
    .info-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
    }
    
    .info-item {
      padding: 12px;
      background: #f8fafc;
      border-radius: 8px;
    }
    
    .info-label {
      font-size: 12px;
      color: #64748b;
      margin-bottom: 3px;
    }
    
    .info-value {
      font-size: 14px;
      font-weight: 600;
      color: #1e293b;
    }
    
    .footer {
      background: #f1f5f9;
      padding: 20px 30px;
      text-align: center;
    }
    
    .footer-disclaimer {
      font-size: 12px;
      color: #64748b;
      margin-bottom: 10px;
      padding: 15px;
      background: #fef3c7;
      border-radius: 8px;
      color: #92400e;
    }
    
    .footer-info {
      font-size: 11px;
      color: #94a3b8;
    }
    
    .image-section {
      text-align: center;
      margin-bottom: 20px;
    }
    
    .image-section img {
      max-width: 300px;
      max-height: 300px;
      border-radius: 12px;
      border: 2px solid #e2e8f0;
    }
    
    @media print {
      body {
        background: white;
        padding: 0;
      }
      
      .report-container {
        box-shadow: none;
        border-radius: 0;
      }
      
      .no-print {
        display: none;
      }
    }
    
    .print-button {
      display: block;
      width: 100%;
      max-width: 800px;
      margin: 20px auto;
      padding: 15px 30px;
      background: linear-gradient(135deg, #8b5cf6, #06b6d4);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 16px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s ease;
    }
    
    .print-button:hover {
      transform: translateY(-2px);
    }
  </style>
</head>
<body>
  <button class="print-button no-print" onclick="window.print()">
    📄 Print / Save as PDF
  </button>
  
  <div class="report-container">
    <div class="header">
      <h1>🩺 DeepBreast AI</h1>
      <p>${data.type === "mammography" ? "Mammography BI-RADS Classification Report" : "Histopathology Analysis Report"}</p>
      <div class="report-id">Report ID: ${reportId} | Generated: ${formatDate(new Date())}</div>
    </div>
    
    <div class="content">
      ${data.imageUrl ? `
      <div class="section image-section">
        <img src="${data.imageUrl}" alt="Analyzed Image" />
      </div>
      ` : ""}
      
      <div class="section">
        <div class="section-title">Analysis Result</div>
        <div class="result-card">
          <div class="result-prediction">${data.prediction}</div>
          <div class="result-confidence">Confidence: ${data.confidence.toFixed(1)}%</div>
          ${data.biradsCategory ? `<span class="birads-badge">${data.biradsCategory}</span>` : ""}
        </div>
      </div>
      
      <div class="section">
        <div class="section-title">Class Probabilities</div>
        <div class="probability-grid">
          ${Object.entries(data.probabilities).map(([key, value]) => `
            <div class="probability-item">
              <div class="probability-label">${key.charAt(0).toUpperCase() + key.slice(1)}</div>
              <div class="probability-value">${(value as number).toFixed(1)}%</div>
            </div>
          `).join("")}
        </div>
      </div>
      
      ${data.recommendation ? `
      <div class="section">
        <div class="section-title">Clinical Recommendation</div>
        <div class="recommendation-card">
          <div class="recommendation-header">
            <div class="recommendation-action">${data.recommendation.action}</div>
            <span class="urgency-badge">${data.recommendation.urgency} Urgency</span>
          </div>
          <p class="recommendation-description">${data.recommendation.description}</p>
          <ul class="next-steps">
            ${data.recommendation.next_steps.map(step => `<li>${step}</li>`).join("")}
          </ul>
        </div>
      </div>
      ` : ""}
      
      <div class="section">
        <div class="section-title">Report Information</div>
        <div class="info-grid">
          <div class="info-item">
            <div class="info-label">Analysis Type</div>
            <div class="info-value">${data.type === "mammography" ? "Mammography" : "Histopathology"}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Date & Time</div>
            <div class="info-value">${formatDate(new Date(data.timestamp))}</div>
          </div>
          ${data.patientId ? `
          <div class="info-item">
            <div class="info-label">Patient ID</div>
            <div class="info-value">${data.patientId}</div>
          </div>
          ` : ""}
          <div class="info-item">
            <div class="info-label">Model</div>
            <div class="info-value">${data.type === "mammography" ? "EfficientNet-B2 (BI-RADS)" : "ResNet18 (Binary)"}</div>
          </div>
        </div>
      </div>
      
      ${data.notes ? `
      <div class="section">
        <div class="section-title">Notes</div>
        <p style="color: #475569;">${data.notes}</p>
      </div>
      ` : ""}
    </div>
    
    <div class="footer">
      <div class="footer-disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This AI analysis is intended to assist healthcare professionals and should not be used as the sole basis for clinical decisions. All findings must be reviewed and confirmed by qualified medical professionals.
      </div>
      <div class="footer-info">
        DeepBreast AI v2.2 | Powered by Deep Learning | © ${new Date().getFullYear()}
      </div>
    </div>
  </div>
  
  <script>
    // Auto-focus print dialog after slight delay
    setTimeout(() => {
      document.querySelector('.print-button').focus();
    }, 500);
  </script>
</body>
</html>
  `;

    printWindow.document.write(htmlContent);
    printWindow.document.close();
};

export default generatePDFReport;
