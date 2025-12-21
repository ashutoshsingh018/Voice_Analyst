"""
Professional PDF Report Generation
EXACT COPY of PDF generation logic from Streamlit
"""

import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_professional_pdf(data):
    """
    Generate professional PDF report
    PRESERVED EXACTLY from Streamlit version
    """
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    
    doc = SimpleDocTemplate(pdf_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title Style (PRESERVED)
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        alignment=1,
        fontSize=24,
        textColor=colors.darkblue
    )
    story.append(Paragraph("Speech Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Updated Table Data (PRESERVED)
    table_data = [
        ["Metric", "Result", "Score"],
        ["Topic", data["topic"]["label"], f"{data['topic']['score']:.1f}%"],
        ["Emotion", data["emotion"]["label"].upper(), f"{data['emotion']['score']:.2f}"],
        ["Sentiment", data["sentiment"]["label"], f"{data['sentiment']['score']:.2f}"],
        ["Toxicity", "Toxic" if data["toxicity"]["score"] > 0.5 else "Safe", 
         f"{data['toxicity']['score']:.2f}"],
        ["Noise", data["noise"][0], f"{data['noise'][1]:.2f}"],
        ["WPM", str(data["wpm"]), "Speed"],
        ["Speaking Time", str(data["speaking_time"]), "Seconds"],
        ["Grammar Score (T5)", str(data["grammar"]["score"]), "AI Index"],
        ["Communication Score", str(data["score"]), "AI Index"],
    ]
    
    # Table Styling (PRESERVED)
    table = Table(table_data, colWidths=[150, 150, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Transcript Section (PRESERVED)
    story.append(Paragraph(
        "<b>Transcript (Language: " + data["lang"] + "):</b>", 
        styles['Heading3']
    ))
    story.append(Paragraph(data["text"], styles["BodyText"]))
    
    # Grammar Correction Section (PRESERVED)
    story.append(Paragraph("<b>T5 Grammar Correction:</b>", styles['Heading3']))
    story.append(Paragraph(data["grammar"]["corrected_text"], styles["BodyText"]))
    story.append(Spacer(1, 12))
    
    # Summary Section (PRESERVED)
    story.append(Paragraph("<b>Summary:</b>", styles['Heading3']))
    story.append(Paragraph(data["summary"], styles["BodyText"]))
    
    # Build PDF
    doc.build(story)
    
    return pdf_file.name