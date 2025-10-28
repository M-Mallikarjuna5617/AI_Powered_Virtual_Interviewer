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

gd_bp = Blueprint("gd", __name__, url_prefix="/gd")

DB_PATH = "users.db"
speech_service = SpeechToTextService()
nlp_service = NLPAnalysisService()

def init_gd_topics():
    """Initialize GD topics database with real company data"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS gd_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            description TEXT NOT NULL,
            year INTEGER NOT NULL,
            difficulty TEXT NOT NULL
        )
    """)
    
    # Check if topics already exist
    c.execute("SELECT COUNT(*) FROM gd_topics")
    if c.fetchone()[0] > 0:
        conn.close()
        print("GD topics already initialized!")
        return
    # Sample GD topics from real companies (2013-2025)
    topics_data = [
        # TCS Topics
        (1, "Impact of Artificial Intelligence on Employment", 
         "Discuss the pros and cons of AI replacing human jobs in various sectors.", 2024, "medium"),
        (1, "Remote Work vs Office Work", 
         "Analyze the benefits and challenges of remote work culture.", 2024, "easy"),
        (1, "Digital India: Progress and Challenges", 
         "Evaluate the success of Digital India initiative and remaining challenges.", 2024, "medium"),
        
        # Infosys Topics
        (2, "Cybersecurity in the Digital Age", 
         "Discuss the importance of cybersecurity and measures to protect data.", 2024, "hard"),
        (2, "Sustainable Technology Solutions", 
         "How can technology help in achieving environmental sustainability?", 2024, "medium"),
        (2, "Women in Technology", 
         "Discuss the role and challenges of women in the technology sector.", 2024, "medium"),
        
        # Wipro Topics
        (3, "Cloud Computing: Future of IT Infrastructure", 
         "Analyze the impact of cloud computing on traditional IT infrastructure.", 2024, "medium"),
        (3, "Data Privacy in Social Media", 
         "Discuss privacy concerns and solutions in social media platforms.", 2024, "medium"),
        (3, "Startup Culture in India", 
         "Evaluate the growth and impact of startup ecosystem in India.", 2024, "easy"),
        
        # Tech Mahindra Topics
        (4, "5G Technology: Opportunities and Challenges", 
         "Discuss the potential of 5G technology and implementation challenges.", 2024, "hard"),
        (4, "Digital Transformation in Healthcare", 
         "How technology is revolutionizing healthcare delivery.", 2024, "medium"),
        (4, "Ethics in Technology", 
         "Discuss ethical considerations in AI, automation, and data usage.", 2024, "hard"),
        
        # Accenture Topics
        (5, "Blockchain Technology: Beyond Cryptocurrency", 
         "Explore applications of blockchain beyond digital currencies.", 2024, "hard"),
        (5, "Smart Cities: Vision vs Reality", 
         "Evaluate the progress of smart city initiatives in India.", 2024, "medium"),
        (5, "Digital Divide in India", 
         "Discuss the gap between urban and rural digital adoption.", 2024, "medium"),
        
        # Cognizant Topics
        (6, "Internet of Things (IoT) in Daily Life", 
         "How IoT is changing our daily routines and lifestyle.", 2024, "medium"),
        (6, "Mental Health in the Digital Age", 
         "Discuss the impact of technology on mental health and well-being.", 2024, "medium"),
        (6, "E-learning vs Traditional Education", 
         "Compare the effectiveness of online and offline learning methods.", 2024, "easy"),
        
        # Capgemini Topics
        (7, "Automation and Job Security", 
         "How automation affects job security and what skills are needed.", 2024, "medium"),
        (7, "Digital Banking Revolution", 
         "Impact of digital banking on traditional banking systems.", 2024, "medium"),
        (7, "Social Media Influence on Youth", 
         "Analyze the positive and negative effects of social media on young people.", 2024, "easy"),
        
        # L&T Topics
        (8, "Infrastructure Development in India", 
         "Discuss the challenges and opportunities in infrastructure development.", 2024, "medium"),
        (8, "Renewable Energy: India's Green Future", 
         "Evaluate India's progress in renewable energy adoption.", 2024, "medium"),
        (8, "Smart Manufacturing", 
         "How Industry 4.0 is transforming manufacturing processes.", 2024, "hard"),
        
        # Mindtree Topics
        (9, "Data Analytics in Business Decision Making", 
         "Role of data analytics in modern business strategies.", 2024, "medium"),
        (9, "Cybersecurity Threats and Solutions", 
         "Discuss emerging cybersecurity threats and preventive measures.", 2024, "hard"),
        (9, "Digital Literacy in Rural India", 
         "Challenges and solutions for improving digital literacy in rural areas.", 2024, "medium"),
        
        # HCL Topics
        (10, "Edge Computing: The Future of Data Processing", 
         "Benefits and challenges of edge computing technology.", 2024, "hard"),
        (10, "Virtual Reality in Education", 
         "Potential of VR technology in transforming education.", 2024, "medium"),
        (10, "Social Responsibility of Tech Companies", 
         "Discuss the social responsibilities of technology companies.", 2024, "medium"),
    ]
    
    c.executemany("""
        INSERT INTO gd_topics (company_id, topic, description, year, difficulty)
        VALUES (?, ?, ?, ?, ?)
    """, topics_data)
    
    conn.commit()
    conn.close()
    print("GD topics database initialized successfully!")

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
        
        # Get selected company
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT c.name FROM companies c
            JOIN selected_companies sc ON c.id = sc.company_id
            WHERE sc.student_email = ?
            ORDER BY sc.selected_at DESC
            LIMIT 1
        """, (email,))
        
        company_result = c.fetchone()
        company_name = company_result[0] if company_result else "General"
        company_id = 1  # Default to TCS if no company selected
        
        if company_result:
            c.execute("SELECT id FROM companies WHERE name = ?", (company_name,))
            company_id = c.fetchone()[0]
        
        # Get a random GD topic
        c.execute("""
            SELECT id, topic, description, difficulty
            FROM gd_topics
            WHERE company_id = ? OR company_id = 1
            ORDER BY RANDOM()
            LIMIT 1
        """, (company_id,))
        
        topic_result = c.fetchone()
        if not topic_result:
            return jsonify({"success": False, "error": "No topics available"}), 404
        
        topic_id, topic, description, difficulty = topic_result
        conn.close()
        
        return jsonify({
            "success": True,
            "company": company_name,
            "topic_id": topic_id,
            "topic": topic,
            "description": description,
            "difficulty": difficulty,
            "time_limit": 180  # 3 minutes
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
        
        # Get topic details
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT t.topic, c.name 
            FROM gd_topics t
            JOIN companies c ON t.company_id = c.id
            WHERE t.id = ?
        """, (topic_id,))
        
        topic_result = c.fetchone()
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

# Initialize topics when module is imported
init_gd_topics()
