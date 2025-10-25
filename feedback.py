"""
Comprehensive Feedback and Report Generation System
Generates detailed reports combining all interview rounds
"""

from flask import Blueprint, jsonify, request, session, render_template, send_file
import sqlite3
import json
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

feedback_bp = Blueprint("feedback", __name__, url_prefix="/feedback")

DB_PATH = "users.db"

@feedback_bp.route("/")
def feedback_home():
    """Feedback dashboard home page"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    return render_template("feedback.html", username=session.get("name"))

@feedback_bp.route("/generate", methods=["POST"])
def generate_comprehensive_feedback():
    """Generate comprehensive feedback report for all rounds"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        # Get all scores from different rounds
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get aptitude scores
        c.execute("""
            SELECT AVG(score), COUNT(*) 
            FROM test_attempts 
            WHERE student_email = ? AND status = 'completed'
        """, (email,))
        aptitude_result = c.fetchone()
        aptitude_score = aptitude_result[0] if aptitude_result[0] else 0
        aptitude_attempts = aptitude_result[1] if aptitude_result[1] else 0
        
        # Get technical scores
        c.execute("""
            SELECT AVG(score), COUNT(*) 
            FROM technical_results 
            WHERE student_email = ?
        """, (email,))
        technical_result = c.fetchone()
        technical_score = technical_result[0] if technical_result[0] else 0
        technical_attempts = technical_result[1] if technical_result[1] else 0
        
        # Get GD scores
        c.execute("""
            SELECT AVG(overall_score), COUNT(*) 
            FROM gd_results 
            WHERE student_email = ?
        """, (email,))
        gd_result = c.fetchone()
        gd_score = gd_result[0] if gd_result[0] else 0
        gd_attempts = gd_result[1] if gd_result[1] else 0
        
        # Get HR scores
        c.execute("""
            SELECT AVG(overall_score), COUNT(*) 
            FROM hr_results 
            WHERE student_email = ?
        """, (email,))
        hr_result = c.fetchone()
        hr_score = hr_result[0] if hr_result[0] else 0
        hr_attempts = hr_result[1] if hr_result[1] else 0
        
        # Get company name
        c.execute("""
            SELECT c.name FROM companies c
            JOIN selected_companies sc ON c.id = sc.company_id
            WHERE sc.student_email = ?
            ORDER BY sc.selected_at DESC
            LIMIT 1
        """, (email,))
        company_result = c.fetchone()
        company_name = company_result[0] if company_result else "General"
        
        # Calculate weighted overall score
        # Aptitude: 30%, Technical: 40%, GD: 15%, HR: 15%
        overall_score = (
            (aptitude_score * 0.30) + 
            (technical_score * 0.40) + 
            (gd_score * 0.15) + 
            (hr_score * 0.15)
        )
        
        # Generate detailed feedback
        feedback_analysis = generate_detailed_feedback(
            aptitude_score, technical_score, gd_score, hr_score, overall_score
        )
        
        # Save comprehensive report
        c.execute("""
            INSERT INTO feedback_reports 
            (student_email, company_name, aptitude_score, technical_score, 
             gd_score, hr_score, overall_score, strengths, improvements, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (email, company_name, aptitude_score, technical_score, 
              gd_score, hr_score, overall_score,
              json.dumps(feedback_analysis['strengths']),
              json.dumps(feedback_analysis['improvements']),
              json.dumps(feedback_analysis['recommendations'])))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "scores": {
                "aptitude": round(aptitude_score, 2),
                "technical": round(technical_score, 2),
                "gd": round(gd_score, 2),
                "hr": round(hr_score, 2),
                "overall": round(overall_score, 2)
            },
            "attempts": {
                "aptitude": aptitude_attempts,
                "technical": technical_attempts,
                "gd": gd_attempts,
                "hr": hr_attempts
            },
            "feedback": feedback_analysis,
            "company": company_name
        })
    
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@feedback_bp.route("/detailed-analysis", methods=["GET"])
def get_detailed_analysis():
    """Get detailed analysis for each round"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get aptitude analysis
        c.execute("""
            SELECT topic, AVG(score), COUNT(*) 
            FROM test_attempts 
            WHERE student_email = ? AND status = 'completed'
            GROUP BY topic
        """, (email,))
        aptitude_analysis = [{"topic": row[0], "avg_score": row[1], "attempts": row[2]} 
                           for row in c.fetchall()]
        
        # Get technical analysis
        c.execute("""
            SELECT domain, AVG(score), COUNT(*) 
            FROM technical_results 
            WHERE student_email = ?
            GROUP BY domain
        """, (email,))
        technical_analysis = [{"domain": row[0], "avg_score": row[1], "attempts": row[2]} 
                            for row in c.fetchall()]
        
        # Get GD analysis
        c.execute("""
            SELECT fluency_score, clarity_score, confidence_score, overall_score
            FROM gd_results 
            WHERE student_email = ?
            ORDER BY submitted_at DESC
            LIMIT 5
        """, (email,))
        gd_analysis = [{"fluency": row[0], "clarity": row[1], "confidence": row[2], "overall": row[3]} 
                      for row in c.fetchall()]
        
        # Get HR analysis
        c.execute("""
            SELECT clarity_score, relevance_score, confidence_score, overall_score
            FROM hr_results 
            WHERE student_email = ?
            ORDER BY submitted_at DESC
            LIMIT 5
        """, (email,))
        hr_analysis = [{"clarity": row[0], "relevance": row[1], "confidence": row[2], "overall": row[3]} 
                      for row in c.fetchall()]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "aptitude_analysis": aptitude_analysis,
            "technical_analysis": technical_analysis,
            "gd_analysis": gd_analysis,
            "hr_analysis": hr_analysis
        })
    
    except Exception as e:
        print(f"Error getting detailed analysis: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@feedback_bp.route("/download-pdf", methods=["GET"])
def download_pdf_report():
    """Download comprehensive PDF report"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        # Get latest report
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT * FROM feedback_reports 
            WHERE student_email = ?
            ORDER BY generated_at DESC
            LIMIT 1
        """, (email,))
        
        report = c.fetchone()
        if not report:
            return jsonify({"success": False, "error": "No report found"}), 404
        
        # Get student details
        c.execute("SELECT * FROM students WHERE email = ?", (email,))
        student = c.fetchone()
        conn.close()
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(report, student)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"Interview_Report_{email}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mimetype='application/pdf'
        )
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def generate_detailed_feedback(aptitude_score, technical_score, gd_score, hr_score, overall_score):
    """Generate detailed feedback analysis"""
    feedback = {
        "strengths": [],
        "improvements": [],
        "recommendations": []
    }
    
    # Overall performance analysis
    if overall_score >= 85:
        feedback["strengths"].append("Excellent overall performance across all rounds")
        feedback["strengths"].append("Strong technical and communication skills")
        feedback["recommendations"].append("Ready for senior technical roles")
        feedback["recommendations"].append("Consider leadership positions")
    elif overall_score >= 70:
        feedback["strengths"].append("Good performance with room for improvement")
        feedback["improvements"].append("Focus on weaker areas identified below")
        feedback["recommendations"].append("Continue practicing and learning")
    else:
        feedback["improvements"].append("Significant improvement needed across multiple areas")
        feedback["recommendations"].append("Consider additional training and practice")
    
    # Round-specific analysis
    if aptitude_score >= 80:
        feedback["strengths"].append("Strong aptitude and logical reasoning skills")
    elif aptitude_score < 60:
        feedback["improvements"].append("Improve aptitude and logical reasoning")
        feedback["recommendations"].append("Practice more aptitude questions and puzzles")
    
    if technical_score >= 80:
        feedback["strengths"].append("Excellent technical and coding skills")
    elif technical_score < 60:
        feedback["improvements"].append("Strengthen technical and coding abilities")
        feedback["recommendations"].append("Practice more coding problems and algorithms")
    
    if gd_score >= 80:
        feedback["strengths"].append("Strong communication and group discussion skills")
    elif gd_score < 60:
        feedback["improvements"].append("Improve communication and group discussion skills")
        feedback["recommendations"].append("Practice public speaking and group discussions")
    
    if hr_score >= 80:
        feedback["strengths"].append("Excellent interpersonal and HR interview skills")
    elif hr_score < 60:
        feedback["improvements"].append("Improve interpersonal and interview skills")
        feedback["recommendations"].append("Practice HR interview questions and scenarios")
    
    # Career recommendations based on scores
    if technical_score > aptitude_score and technical_score > 75:
        feedback["recommendations"].append("Consider technical specialist roles")
    elif gd_score > hr_score and gd_score > 75:
        feedback["recommendations"].append("Consider roles requiring strong communication")
    elif hr_score > technical_score and hr_score > 75:
        feedback["recommendations"].append("Consider management and leadership roles")
    
    return feedback

def generate_pdf_report(report, student):
    """Generate PDF report using ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("AI-Powered Virtual Interview Report", title_style))
    story.append(Spacer(1, 20))
    
    # Student Information
    story.append(Paragraph("Student Information", styles['Heading2']))
    student_info = [
        ["Name:", student[2] if student else "N/A"],
        ["Email:", student[1] if student else "N/A"],
        ["CGPA:", str(student[3]) if student else "N/A"],
        ["Graduation Year:", str(student[4]) if student else "N/A"],
        ["Skills:", student[5] if student else "N/A"]
    ]
    
    student_table = Table(student_info, colWidths=[2*inch, 4*inch])
    student_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
    ]))
    story.append(student_table)
    story.append(Spacer(1, 20))
    
    # Scores Summary
    story.append(Paragraph("Performance Summary", styles['Heading2']))
    scores_data = [
        ["Round", "Score", "Weight", "Weighted Score"],
        ["Aptitude", f"{report[3]:.1f}%", "30%", f"{report[3] * 0.30:.1f}%"],
        ["Technical", f"{report[4]:.1f}%", "40%", f"{report[4] * 0.40:.1f}%"],
        ["Group Discussion", f"{report[5]:.1f}%", "15%", f"{report[5] * 0.15:.1f}%"],
        ["HR Interview", f"{report[6]:.1f}%", "15%", f"{report[6] * 0.15:.1f}%"],
        ["", "", "Total", f"{report[7]:.1f}%"]
    ]
    
    scores_table = Table(scores_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
    scores_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(scores_table)
    story.append(Spacer(1, 20))
    
    # Feedback
    story.append(Paragraph("Detailed Feedback", styles['Heading2']))
    
    strengths = json.loads(report[8]) if report[8] else []
    improvements = json.loads(report[9]) if report[9] else []
    recommendations = json.loads(report[10]) if report[10] else []
    
    if strengths:
        story.append(Paragraph("Strengths:", styles['Heading3']))
        for strength in strengths:
            story.append(Paragraph(f"• {strength}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    if improvements:
        story.append(Paragraph("Areas for Improvement:", styles['Heading3']))
        for improvement in improvements:
            story.append(Paragraph(f"• {improvement}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    if recommendations:
        story.append(Paragraph("Recommendations:", styles['Heading3']))
        for recommendation in recommendations:
            story.append(Paragraph(f"• {recommendation}", styles['Normal']))
        story.append(Spacer(1, 10))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Report Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                          styles['Normal']))
    story.append(Paragraph("AI-Powered Virtual Interview System", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@feedback_bp.route("/history", methods=["GET"])
def get_feedback_history():
    """Get feedback report history"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT company_name, aptitude_score, technical_score, gd_score, 
                   hr_score, overall_score, generated_at
            FROM feedback_reports 
            WHERE student_email = ?
            ORDER BY generated_at DESC
        """, (email,))
        
        history = []
        for row in c.fetchall():
            history.append({
                "company": row[0],
                "aptitude_score": row[1],
                "technical_score": row[2],
                "gd_score": row[3],
                "hr_score": row[4],
                "overall_score": row[5],
                "generated_at": row[6]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "history": history
        })
    
    except Exception as e:
        print(f"Error getting feedback history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
