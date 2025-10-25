"""
HR Interview Module
AI-powered HR interview with personalized questions and evaluation
"""

from flask import Blueprint, jsonify, request, session, render_template
import sqlite3
import json
import time
import random
from datetime import datetime
from ai_service import HRInterviewService

hr_bp = Blueprint("hr", __name__, url_prefix="/hr")

DB_PATH = "users.db"
hr_service = HRInterviewService()

def init_hr_questions():
    """Initialize HR questions database with real company data"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            category TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            year INTEGER NOT NULL
        )
    """)
    
    # Check if questions already exist
    c.execute("SELECT COUNT(*) FROM hr_questions")
    if c.fetchone()[0] > 0:
        conn.close()
        print("✅ HR questions already initialized!")
        return
    
    # Sample HR questions from real companies (2013-2025)
    questions_data = [
        # TCS Questions
        (1, "Tell me about yourself and your background.", "Introduction", "easy", 2024),
        (1, "Why do you want to work at TCS?", "Motivation", "easy", 2024),
        (1, "Describe a challenging project you worked on.", "Experience", "medium", 2024),
        (1, "How do you handle pressure and deadlines?", "Behavioral", "medium", 2024),
        (1, "Where do you see yourself in 5 years?", "Career Goals", "easy", 2024),
        
        # Infosys Questions
        (2, "What motivates you to perform well?", "Motivation", "easy", 2024),
        (2, "Describe a time when you had to work in a team.", "Teamwork", "medium", 2024),
        (2, "How do you stay updated with technology trends?", "Learning", "medium", 2024),
        (2, "What is your greatest strength and weakness?", "Self Assessment", "medium", 2024),
        (2, "How do you handle conflicts in a team?", "Conflict Resolution", "hard", 2024),
        
        # Wipro Questions
        (3, "Why should we hire you?", "Value Proposition", "medium", 2024),
        (3, "Describe a situation where you had to learn something new quickly.", "Adaptability", "medium", 2024),
        (3, "How do you prioritize your tasks?", "Time Management", "easy", 2024),
        (3, "What do you know about our company?", "Company Knowledge", "easy", 2024),
        (3, "How do you handle criticism?", "Resilience", "medium", 2024),
        
        # Tech Mahindra Questions
        (4, "Tell me about a time you failed and what you learned.", "Learning from Failure", "hard", 2024),
        (4, "How do you ensure quality in your work?", "Quality Focus", "medium", 2024),
        (4, "Describe your ideal work environment.", "Work Preferences", "easy", 2024),
        (4, "How do you handle multiple projects simultaneously?", "Multitasking", "medium", 2024),
        (4, "What makes you different from other candidates?", "Differentiation", "hard", 2024),
        
        # Accenture Questions
        (5, "How do you approach problem-solving?", "Problem Solving", "medium", 2024),
        (5, "Describe a time when you had to convince someone.", "Influence", "hard", 2024),
        (5, "How do you handle ambiguity in projects?", "Adaptability", "medium", 2024),
        (5, "What is your approach to continuous learning?", "Learning", "medium", 2024),
        (5, "How do you measure success in your work?", "Success Metrics", "medium", 2024),
        
        # Cognizant Questions
        (6, "Describe a time when you had to work with difficult people.", "Interpersonal Skills", "hard", 2024),
        (6, "How do you handle tight deadlines?", "Time Management", "medium", 2024),
        (6, "What role do you prefer in a team?", "Team Role", "easy", 2024),
        (6, "How do you stay motivated during challenging times?", "Motivation", "medium", 2024),
        (6, "Describe your leadership style.", "Leadership", "hard", 2024),
        
        # Capgemini Questions
        (7, "How do you handle feedback from supervisors?", "Feedback Reception", "medium", 2024),
        (7, "Describe a time when you had to adapt to change.", "Change Management", "medium", 2024),
        (7, "How do you ensure effective communication?", "Communication", "medium", 2024),
        (7, "What do you do when you don't know something?", "Learning Attitude", "easy", 2024),
        (7, "How do you balance work and personal life?", "Work Life Balance", "medium", 2024),
        
        # L&T Questions
        (8, "Describe a time when you had to make a difficult decision.", "Decision Making", "hard", 2024),
        (8, "How do you handle stress in the workplace?", "Stress Management", "medium", 2024),
        (8, "What is your approach to innovation?", "Innovation", "medium", 2024),
        (8, "How do you build relationships with colleagues?", "Relationship Building", "medium", 2024),
        (8, "Describe a time when you had to learn from a mistake.", "Learning from Mistakes", "medium", 2024),
        
        # Mindtree Questions
        (9, "How do you approach mentoring others?", "Mentoring", "hard", 2024),
        (9, "Describe a time when you had to work with limited resources.", "Resource Management", "hard", 2024),
        (9, "How do you handle ethical dilemmas?", "Ethics", "hard", 2024),
        (9, "What is your approach to risk-taking?", "Risk Management", "medium", 2024),
        (9, "How do you ensure customer satisfaction?", "Customer Focus", "medium", 2024),
        
        # HCL Questions
        (10, "Describe a time when you had to lead a project.", "Leadership", "hard", 2024),
        (10, "How do you handle competing priorities?", "Priority Management", "medium", 2024),
        (10, "What is your approach to continuous improvement?", "Process Improvement", "medium", 2024),
        (10, "How do you handle technology changes?", "Technology Adaptation", "medium", 2024),
        (10, "Describe your approach to teamwork.", "Teamwork", "medium", 2024),
    ]
    
    c.executemany("""
        INSERT INTO hr_questions (company_id, question, category, difficulty, year)
        VALUES (?, ?, ?, ?, ?)
    """, questions_data)
    
    conn.commit()
    conn.close()
    print("✅ HR questions database initialized successfully!")
@hr_bp.route("/")
def hr_home():
    """HR interview home page"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    return render_template("hr.html", username=session.get("name"))

@hr_bp.route("/start", methods=["POST"])
def start_hr_interview():
    """Start an HR interview session"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        # Get student profile
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM students WHERE email = ?", (email,))
        student = c.fetchone()
        
        if not student:
            return jsonify({"success": False, "error": "Student profile not found"}), 404
        
        # Get selected company
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
        
        # Get 5 random HR questions
        c.execute("""
            SELECT id, question, category, difficulty
            FROM hr_questions
            WHERE company_id = ? OR company_id = 1
            ORDER BY RANDOM()
            LIMIT 5
        """, (company_id,))
        
        questions = []
        for row in c.fetchall():
            questions.append({
                "id": row[0],
                "question": row[1],
                "category": row[2],
                "difficulty": row[3]
            })
        
        conn.close()
        
        # Create student profile for AI
        student_profile = {
            "name": student[2],
            "cgpa": student[3],
            "skills": student[5],
            "graduation_year": student[4]
        }
        
        return jsonify({
            "success": True,
            "company": company_name,
            "questions": questions,
            "student_profile": student_profile,
            "time_limit": 30  # 30 minutes
        })
    
    except Exception as e:
        print(f"Error starting HR interview: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@hr_bp.route("/generate-question", methods=["POST"])
def generate_ai_question():
    """Generate AI-powered personalized HR question"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        student_profile = data.get('student_profile', {})
        company_name = data.get('company_name', 'General')
        previous_answers = data.get('previous_answers', [])
        
        # Generate AI question
        ai_question = hr_service.generate_question(student_profile, previous_answers, company_name)
        
        return jsonify({
            "success": True,
            "question": ai_question.get('question', 'Tell me about yourself.'),
            "generated_at": ai_question.get('generated_at', time.time())
        })
    
    except Exception as e:
        print(f"Error generating AI question: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@hr_bp.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    """Evaluate student's answer using AI"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        question = data.get('question', '')
        answer = data.get('answer', '')
        question_id = data.get('question_id')
        
        if not answer.strip():
            return jsonify({"success": False, "error": "No answer provided"}), 400
        
        # Evaluate answer using AI
        evaluation = hr_service.evaluate_answer(question, answer)
        
        # Get company name
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT c.name FROM companies c
            JOIN selected_companies sc ON c.id = sc.company_id
            WHERE sc.student_email = ?
            ORDER BY sc.selected_at DESC
            LIMIT 1
        """, (session["email"],))
        
        company_result = c.fetchone()
        company_name = company_result[0] if company_result else "General"
        
        # Save result
        c.execute("""
            INSERT INTO hr_results 
            (student_email, company_name, question_id, answer, clarity_score, 
             relevance_score, confidence_score, overall_score, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session["email"], company_name, question_id, answer,
              evaluation.get('clarity_score', 75),
              evaluation.get('relevance_score', 75),
              evaluation.get('confidence_score', 75),
              evaluation.get('overall_score', 75),
              json.dumps(evaluation.get('feedback', {}))))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "scores": {
                "clarity": evaluation.get('clarity_score', 75),
                "relevance": evaluation.get('relevance_score', 75),
                "confidence": evaluation.get('confidence_score', 75),
                "overall": evaluation.get('overall_score', 75)
            },
            "feedback": evaluation.get('feedback', {}),
            "strengths": evaluation.get('strengths', []),
            "improvements": evaluation.get('improvements', [])
        })
    
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@hr_bp.route("/submit", methods=["POST"])
def submit_hr_interview():
    """Submit complete HR interview"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        email = session["email"]
        answers = data.get('answers', [])
        
        # Calculate overall score
        total_score = 0
        total_questions = len(answers)
        
        for answer in answers:
            total_score += answer.get('overall_score', 0)
        
        overall_score = (total_score / total_questions) if total_questions > 0 else 0
        
        # Get company name
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
        
        # Generate comprehensive feedback
        feedback = generate_hr_feedback(answers, overall_score)
        
        conn.close()
        
        return jsonify({
            "success": True,
            "overall_score": round(overall_score, 2),
            "total_questions": total_questions,
            "feedback": feedback,
            "company": company_name
        })
    
    except Exception as e:
        print(f"Error submitting HR interview: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@hr_bp.route("/history", methods=["GET"])
def get_hr_history():
    """Get student's HR interview history"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT hr.overall_score, hr.clarity_score, hr.relevance_score, 
                   hr.confidence_score, hr.submitted_at, hq.question, c.name
            FROM hr_results hr
            JOIN hr_questions hq ON hr.question_id = hq.id
            JOIN companies c ON hq.company_id = c.id
            WHERE hr.student_email = ?
            ORDER BY hr.submitted_at DESC
            LIMIT 10
        """, (email,))
        
        history = []
        for row in c.fetchall():
            history.append({
                "overall_score": row[0],
                "clarity_score": row[1],
                "relevance_score": row[2],
                "confidence_score": row[3],
                "submitted_at": row[4],
                "question": row[5],
                "company": row[6]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "history": history
        })
    
    except Exception as e:
        print(f"Error getting HR history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def generate_hr_feedback(answers, overall_score):
    """Generate comprehensive HR interview feedback"""
    feedback = {
        "overall_score": overall_score,
        "strengths": [],
        "improvements": [],
        "recommendations": []
    }
    
    if overall_score >= 80:
        feedback["strengths"].append("Excellent communication skills")
        feedback["strengths"].append("Strong professional presence")
        feedback["recommendations"].append("Ready for senior positions")
    elif overall_score >= 60:
        feedback["strengths"].append("Good communication foundation")
        feedback["improvements"].append("Work on confidence and clarity")
        feedback["recommendations"].append("Practice more interview scenarios")
    else:
        feedback["improvements"].append("Focus on improving communication skills")
        feedback["improvements"].append("Practice answering common HR questions")
        feedback["recommendations"].append("Consider interview coaching or practice sessions")
    
    # Analyze specific aspects
    clarity_scores = [a.get('clarity_score', 0) for a in answers]
    relevance_scores = [a.get('relevance_score', 0) for a in answers]
    confidence_scores = [a.get('confidence_score', 0) for a in answers]
    
    if clarity_scores and sum(clarity_scores) / len(clarity_scores) < 60:
        feedback["improvements"].append("Work on clear and articulate responses")
    if relevance_scores and sum(relevance_scores) / len(relevance_scores) < 60:
        feedback["improvements"].append("Focus on providing relevant and specific examples")
    if confidence_scores and sum(confidence_scores) / len(confidence_scores) < 60:
        feedback["improvements"].append("Build confidence in expressing your thoughts")
    
    return feedback

# Initialize questions when module is imported
init_hr_questions()
