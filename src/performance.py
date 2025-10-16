import streamlit as st
import json
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import datetime

# ==========================================================
# 📈 Model Performance Panel (Interactive + PDF Export)
# ==========================================================
def run_performance():
    st.title("📈 Model Performance Dashboard (Interactive)")
    st.write("""
    Explore the model’s **training and evaluation metrics** interactively — 
    and export the complete summary as a professional PDF report.
    """)

    history_path = "models/train_history.json"
    eval_path = "models/eval_results.json"

    if not os.path.exists(history_path):
        st.warning("⚠️ No training log found. Please run `train_model.py` first.")
        return
    if not os.path.exists(eval_path):
        st.warning("⚠️ Evaluation results not found. Please run `evaluate_model.py`.")
        return

    # ------------------------------------------------------
    # 🔹 Load Training Logs
    # ------------------------------------------------------
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history["epoch"]
    train_acc = [a * 100 for a in history["train_acc"]]
    val_acc = [a * 100 for a in history["val_acc"]]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    # ------------------------------------------------------
    # 🔹 Load Evaluation Results
    # ------------------------------------------------------
    with open(eval_path, "r") as f:
        results = json.load(f)

    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    cm = np.array(results["confusion_matrix"])

    # ------------------------------------------------------
    # 📊 Plotly Charts (Interactive)
    # ------------------------------------------------------
    st.subheader("📊 Training Progress")

    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers',
                                 name='Train Accuracy', line=dict(color='#00CC96', width=3)))
    acc_fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers',
                                 name='Validation Accuracy', line=dict(color='#636EFA', width=3)))
    acc_fig.update_layout(title="Accuracy over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy (%)",
                          template="plotly_dark", height=400)
    st.plotly_chart(acc_fig, use_container_width=True)

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers',
                                  name='Train Loss', line=dict(color='#EF553B', width=3)))
    loss_fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers',
                                  name='Validation Loss', line=dict(color='#FFA15A', width=3)))
    loss_fig.update_layout(title="Loss over Epochs", xaxis_title="Epoch", yaxis_title="Loss",
                           template="plotly_dark", height=400)
    st.plotly_chart(loss_fig, use_container_width=True)

    st.divider()

    # ------------------------------------------------------
    # 🧠 Evaluation Metrics (Interactive)
    # ------------------------------------------------------
    st.subheader("📈 Evaluation Metrics (Test Set)")

    metrics_fig = px.bar(
        x=["Precision", "Recall", "F1-score"],
        y=[precision * 100, recall * 100, f1 * 100],
        text=[f"{precision*100:.2f}%", f"{recall*100:.2f}%", f"{f1*100:.2f}%"],
        color=["Precision", "Recall", "F1-score"],
        color_discrete_sequence=["#00CC96", "#AB63FA", "#FFA15A"],
        template="plotly_dark"
    )
    metrics_fig.update_traces(textposition="outside")
    metrics_fig.update_layout(yaxis_title="Score (%)", xaxis_title="Metric", height=400)
    st.plotly_chart(metrics_fig, use_container_width=True)

    st.divider()

    # ------------------------------------------------------
    # 🔍 Confusion Matrix (Interactive)
    # ------------------------------------------------------
    st.subheader("🔍 Confusion Matrix (Test Set)")

    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred: Benign", "Pred: Malignant"],
        y=["True: Benign", "True: Malignant"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Purples",
        showscale=False
    ))
    cm_fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        height=400,
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    st.success("✅ Interactive performance dashboard loaded successfully!")

    # ======================================================
    # 📄 Generate PDF Report
    # ======================================================
    st.divider()
    st.subheader("📄 Download Performance Report")

    if st.button("🧾 Generate PDF Report"):
        pdf_path = generate_pdf_report(train_acc, val_acc, train_loss, val_loss,
                                       precision, recall, f1, cm)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Report as PDF",
                data=f,
                file_name="DeepBreast_Model_Report.pdf",
                mime="application/pdf"
            )

    st.caption("🧬 DeepBreast: AI-Based Breast Cancer Detection — Interactive Evaluation & Reporting Module")


# ==========================================================
# 🧾 PDF Oluşturma Fonksiyonu
# ==========================================================
def generate_pdf_report(train_acc, val_acc, train_loss, val_loss,
                        precision, recall, f1, cm):

    os.makedirs("reports", exist_ok=True)
    pdf_path = os.path.join("reports", "DeepBreast_Model_Report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    title = Paragraph("<b>DeepBreast: AI-Based Breast Cancer Detection</b>", styles['Title'])
    date = Paragraph(f"<b>Report Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                     styles['Normal'])
    flowables.extend([title, Spacer(1, 12), date, Spacer(1, 24)])

    # Summary Table
    summary_data = [
        ["Metric", "Value"],
        ["Best Validation Accuracy", f"{max(val_acc):.2f}%"],
        ["Final Training Loss", f"{train_loss[-1]:.4f}"],
        ["Final Validation Loss", f"{val_loss[-1]:.4f}"],
        ["Precision", f"{precision*100:.2f}%"],
        ["Recall", f"{recall*100:.2f}%"],
        ["F1-score", f"{f1*100:.2f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))
    flowables.extend([Paragraph("<b>Performance Summary</b>", styles['Heading2']),
                      Spacer(1, 12), summary_table, Spacer(1, 24)])

    # Confusion Matrix Table
    cm_data = [["", "Pred: Benign", "Pred: Malignant"],
               ["True: Benign", cm[0][0], cm[0][1]],
               ["True: Malignant", cm[1][0], cm[1][1]]]
    cm_table = Table(cm_data, colWidths=[150, 150, 150])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    flowables.extend([Paragraph("<b>Confusion Matrix</b>", styles['Heading2']),
                      Spacer(1, 12), cm_table])

    doc.build(flowables)
    return pdf_path
