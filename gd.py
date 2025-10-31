"""
Group Discussion (GD) Module
Handles GD topics, speech recognition, and AI-powered evaluation
"""

from flask import Blueprint, jsonify, request, session, render_template
import sqlite3
import json
import time
import random
import re
from datetime import datetime
from ai_service import SpeechToTextService, NLPAnalysisService
from utils import get_selected_company

gd_bp = Blueprint("gd", __name__, url_prefix="/gd")

DB_PATH = "users.db"
QUESTIONS_DB_PATH = "questions.db"
speech_service = SpeechToTextService()
nlp_service = NLPAnalysisService()

def init_gd_topics():
    """GD topics are managed via questions.db from JSON datasets; no-op here."""
    return

@gd_bp.route("/")
def gd_home():
    """GD simulation home page"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    return render_template("gd.html", username=session.get("name"))

@gd_bp.route("/start", methods=["POST"])
def start_gd_session():
    """Start a GD session with a random topic"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        selected = get_selected_company(email)
        company_name = selected["name"] if selected else None

        # Map to company_id in questions.db
        qconn = sqlite3.connect(QUESTIONS_DB_PATH)
        qc = qconn.cursor()
        company_id = None
        if company_name:
            qc.execute("SELECT id FROM companies WHERE lower(trim(name)) = lower(trim(?))", (company_name,))
            cid_row = qc.fetchone()
            company_id = cid_row[0] if cid_row else None

        if not company_id:
            qconn.close()
            return jsonify({"success": False, "error": "Please select a company first"}), 400

        # Get a random GD topic for this company from questions.db
        qc.execute(
            """
            SELECT id, topic, description, time_limit
            FROM gd_topics
            WHERE company_id = ?
            ORDER BY RANDOM()
            LIMIT 1
            """,
            (company_id,)
        )

        topic_result = qc.fetchone()
        qconn.close()
        if not topic_result:
            return jsonify({"success": False, "error": "No topics available"}), 404
        
        topic_id, topic, description, time_limit = topic_result
        
        return jsonify({
            "success": True,
            "company": company_name,
            "topic_id": topic_id,
            "topic": topic,
            "description": description,
            "difficulty": "Medium",
            "time_limit": time_limit if time_limit else 180
        })
    
    except Exception as e:
        print(f"Error starting GD session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@gd_bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Transcribe audio to text using speech recognition"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        # In a real implementation, you would handle file upload here
        # For now, we'll simulate transcription
        data = request.json
        audio_data = data.get('audio_data', '')
        
        # Simulate transcription (in real app, use speech_service.transcribe_audio)
        # For demo purposes, return a sample transcript
        sample_transcripts = [
            "I believe that artificial intelligence will revolutionize the way we work. However, we need to ensure that it complements human skills rather than replacing them entirely. The key is to focus on upskilling and reskilling the workforce.",
            "From my perspective, remote work has both advantages and disadvantages. While it offers flexibility and work-life balance, it can also lead to isolation and communication challenges. Companies need to find the right balance.",
            "Digital transformation is crucial for India's growth. We need to bridge the digital divide between urban and rural areas and ensure that technology benefits everyone, not just the privileged few."
        ]
        
        transcript = random.choice(sample_transcripts)
        
        return jsonify({
            "success": True,
            "transcript": transcript,
            "confidence": random.uniform(0.85, 0.95)
        })
    
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@gd_bp.route("/evaluate", methods=["POST"])
def evaluate_gd_performance():
    """Evaluate GD performance using AI analysis"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        transcript = data.get('transcript', '')
        topic_id = data.get('topic_id')
        duration = data.get('duration', 0)
        
        if not transcript.strip():
            return jsonify({"success": False, "error": "No transcript provided"}), 400
        
        # Analyze transcript using NLP
        analysis_result = nlp_service.analyze_text(transcript)
        
        # Calculate scores with better logic
        transcript_length = len(transcript.split())
        word_pause_ratio = duration / max(transcript_length, 1)
        
        fluency_score = min(100, max(40, 70 + (transcript_length * 0.5)))
        clarity_score = min(100, max(50, 75 + (min(transcript_length, 100) / 2)))
        confidence_score = min(100, max(50, 65 + (transcript_length / 3)))
        
        # Deduct for short responses
        if transcript_length < 50:
            fluency_score -= 15
            clarity_score -= 10
            confidence_score -= 15
        
        overall_score = (fluency_score + clarity_score + confidence_score) / 3
        
        # Get topic details from questions.db
        qconn = sqlite3.connect(QUESTIONS_DB_PATH)
        qc = qconn.cursor()
        qc.execute(
            """
            SELECT t.topic, c.name
            FROM gd_topics t
            JOIN companies c ON t.company_id = c.id
            WHERE t.id = ?
            """,
            (topic_id,)
        )
        topic_result = qc.fetchone()
        qconn.close()
        topic_name = topic_result[0] if topic_result else "General Topic"
        company_name = topic_result[1] if topic_result else "General"
        
        c.execute("""
            INSERT INTO gd_results 
            (student_email, company_name, topic_id, transcript, fluency_score, 
             clarity_score, confidence_score, overall_score, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["email"], company_name, topic_id, transcript,
            round(fluency_score, 2), round(clarity_score, 2), 
            round(confidence_score, 2), round(overall_score, 2),
            json.dumps(analysis_result.get('feedback', {}))
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "scores": {
                "fluency": round(fluency_score, 2),
                "clarity": round(clarity_score, 2),
                "confidence": round(confidence_score, 2),
                "overall": round(overall_score, 2)
            },
            "feedback": analysis_result.get('feedback', {}),
            "topic": topic_name,
            "company": company_name,
            "can_continue": True
        })
    
    except Exception as e:
        print(f"Error evaluating GD performance: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@gd_bp.route("/history", methods=["GET"])
def get_gd_history():
    """Get student's GD performance history"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT gr.overall_score, gr.fluency_score, gr.clarity_score, 
                   gr.confidence_score, gr.submitted_at, t.topic, c.name
            FROM gd_results gr
            JOIN gd_topics t ON gr.topic_id = t.id
            JOIN companies c ON t.company_id = c.id
            WHERE gr.student_email = ?
            ORDER BY gr.submitted_at DESC
            LIMIT 10
        """, (email,))
        
        history = []
        for row in c.fetchall():
            history.append({
                "overall_score": row[0],
                "fluency_score": row[1],
                "clarity_score": row[2],
                "confidence_score": row[3],
                "submitted_at": row[4],
                "topic": row[5],
                "company": row[6]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "history": history
        })
    
    except Exception as e:
        print(f"Error getting GD history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def generate_gd_feedback(scores, transcript):
    """Generate personalized feedback for GD performance"""
    feedback = {
        "strengths": [],
        "improvements": [],
        "recommendations": []
    }
    
    overall_score = scores.get('overall', 0)
    
    if overall_score >= 80:
        feedback["strengths"].append("Excellent communication skills")
        feedback["strengths"].append("Clear and confident expression")
        feedback["recommendations"].append("Ready for leadership roles")
    elif overall_score >= 60:
        feedback["strengths"].append("Good communication foundation")
        feedback["improvements"].append("Work on confidence and clarity")
        feedback["recommendations"].append("Practice more group discussions")
    else:
        feedback["improvements"].append("Focus on improving communication skills")
        feedback["improvements"].append("Practice speaking clearly and confidently")
        feedback["recommendations"].append("Join public speaking clubs or courses")
    
    # Analyze specific aspects
    if scores.get('fluency', 0) < 60:
        feedback["improvements"].append("Work on speech fluency and flow")
    if scores.get('clarity', 0) < 60:
        feedback["improvements"].append("Improve pronunciation and articulation")
    if scores.get('confidence', 0) < 60:
        feedback["improvements"].append("Build confidence in expressing opinions")
    
    return feedback

# Topics are sourced from questions.db datasets
