"""
Improved HR Interview Module - COMPLETE WORKING VERSION
- 10 company-specific questions
- 1 minute per question
- Real-time AI evaluation
- Video + Audio monitoring
"""

from flask import Blueprint, jsonify, request, session, render_template
import sqlite3
import json
from datetime import datetime
from ai_service import EnhancedNLPAnalysisService

hr_bp = Blueprint("hr", __name__, url_prefix="/hr")
DB_PATH = "users.db"
nlp_service = EnhancedNLPAnalysisService()

@hr_bp.route("/")
def hr_home():
    """HR interview home page"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    return render_template("hr.html", username=session.get("name"))

@hr_bp.route("/start", methods=["POST"])
def start_hr_interview():
    """Start HR interview with 10 company-specific questions"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        # Get selected company
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT c.id, c.name FROM companies c
            JOIN selected_companies sc ON c.id = sc.company_id
            WHERE sc.student_email = ?
            ORDER BY sc.selected_at DESC
            LIMIT 1
        """, (email,))
        
        company_result = c.fetchone()
        if not company_result:
            return jsonify({"success": False, "error": "Please select a company first"}), 400
        
        company_id, company_name = company_result
        
        # Get 10 random HR questions
        c.execute("""
            SELECT id, question, category, ideal_answer_points, difficulty, time_limit
            FROM hr_question_bank
            WHERE company_id = ?
            ORDER BY RANDOM()
            LIMIT 10
        """, (company_id,))
        
        questions = []
        for row in c.fetchall():
            ideal_points = json.loads(row[3]) if row[3] else []
            questions.append({
                "id": row[0],
                "question": row[1],
                "category": row[2],
                "ideal_points": ideal_points,
                "difficulty": row[4],
                "time_limit": row[5]
            })
        
        if len(questions) < 10:
            return jsonify({
                "success": False,
                "error": f"Insufficient questions for {company_name}"
            }), 400
        
        # Create HR attempt
        c.execute("""
            INSERT INTO hr_attempts (
                student_email, company_name, total_questions, status
            ) VALUES (?, ?, 10, 'in_progress')
        """, (email, company_name))
        
        attempt_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "attempt_id": attempt_id,
            "company": company_name,
            "questions": questions,
            "time_per_question": 60,  # 1 minute per question
            "total_time": 600  # 10 minutes total
        })
    
    except Exception as e:
        print(f"Error starting HR interview: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@hr_bp.route("/submit-answer", methods=["POST"])
def submit_answer():
    """Submit single answer with AI evaluation"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        attempt_id = data.get('attempt_id')
        question_id = data.get('question_id')
        answer = data.get('answer', '').strip()
        time_spent = data.get('time_spent', 0)
        
        if not answer or len(answer) < 20:
            return jsonify({
                "success": False,
                "error": "Answer too short. Please provide a detailed response."
            }), 400
        
        # AI Evaluation
        analysis = nlp_service.analyze_text(answer)
        
        clarity_score = analysis.get('clarity_score', 0)
        confidence_score = analysis.get('confidence_score', 0)
        relevance_score = analysis.get('relevance_score', 0)
        grammar_score = analysis.get('grammar_score', 0)
        
        # Overall score for this answer
        answer_score = (
            clarity_score * 0.30 +
            confidence_score * 0.25 +
            relevance_score * 0.30 +
            grammar_score * 0.15
        )
        
        # Save answer
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO hr_answers (
                attempt_id, question_id, answer, clarity_score,
                confidence_score, relevance_score, grammar_score,
                answer_score, time_spent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (attempt_id, question_id, answer, clarity_score,
              confidence_score, relevance_score, grammar_score,
              answer_score, time_spent))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "scores": {
                "clarity": round(clarity_score, 2),
                "confidence": round(confidence_score, 2),
                "relevance": round(relevance_score, 2),
                "grammar": round(grammar_score, 2),
                "answer_score": round(answer_score, 2)
            },
            "feedback": analysis.get('feedback', ''),
            "word_count": analysis.get('word_count', 0)
        })
    
    except Exception as e:
        print(f"Error submitting answer: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@hr_bp.route("/submit", methods=["POST"])
def submit_hr_interview():
    """Complete HR interview and calculate final score"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        attempt_id = data.get('attempt_id')
        total_time = data.get('total_time', 0)
        
        if not attempt_id:
            return jsonify({"success": False, "error": "Invalid attempt"}), 400
        
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Verify attempt
        c.execute("""
            SELECT student_email, company_name, status
            FROM hr_attempts
            WHERE id = ?
        """, (attempt_id,))
        
        attempt = c.fetchone()
        if not attempt or attempt[0] != email:
            return jsonify({"success": False, "error": "Invalid attempt"}), 403
        
        if attempt[2] == 'completed':
            return jsonify({"success": False, "error": "Already submitted"}), 400
        
        # Calculate overall scores
        c.execute("""
            SELECT AVG(clarity_score), AVG(confidence_score),
                   AVG(relevance_score), AVG(grammar_score), AVG(answer_score),
                   COUNT(*)
            FROM hr_answers
            WHERE attempt_id = ?
        """, (attempt_id,))
        
        scores = c.fetchone()
        if not scores or scores[5] < 10:
            return jsonify({
                "success": False,
                "error": "Please answer all 10 questions"
            }), 400
        
        avg_clarity = scores[0]
        avg_confidence = scores[1]
        avg_relevance = scores[2]
        avg_grammar = scores[3]
        overall_score = scores[4]
        
        # Update attempt
        c.execute("""
            UPDATE hr_attempts
            SET clarity_score = ?,
                confidence_score = ?,
                relevance_score = ?,
                grammar_score = ?,
                overall_score = ?,
                total_time = ?,
                status = 'completed',
                completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (avg_clarity, avg_confidence, avg_relevance, avg_grammar,
              overall_score, total_time, attempt_id))
        
        conn.commit()
        conn.close()
        
        # Generate feedback
        feedback = generate_hr_feedback(overall_score, avg_clarity, avg_confidence, avg_relevance)
        
        return jsonify({
            "success": True,
            "scores": {
                "clarity": round(avg_clarity, 2),
                "confidence": round(avg_confidence, 2),
                "relevance": round(avg_relevance, 2),
                "grammar": round(avg_grammar, 2),
                "overall": round(overall_score, 2)
            },
            "feedback": feedback,
            "message": "HR interview completed successfully!"
        })
    
    except Exception as e:
        print(f"Error submitting HR interview: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def generate_hr_feedback(overall, clarity, confidence, relevance):
    """Generate feedback based on scores"""
    feedback = {
        "performance": "",
        "strengths": [],
        "improvements": [],
        "recommendations": []
    }
    
    if overall >= 80:
        feedback["performance"] = "Excellent interview performance!"
        feedback["strengths"].extend([
            "Clear and articulate communication",
            "High confidence in responses",
            "Relevant and focused answers"
        ])
        feedback["recommendations"].append("Ready for actual interviews")
    elif overall >= 60:
        feedback["performance"] = "Good performance with room for improvement"
        feedback["strengths"].append("Solid communication foundation")
        if clarity < 70:
            feedback["improvements"].append("Work on clarity and articulation")
        if confidence < 70:
            feedback["improvements"].append("Build confidence in delivery")
        feedback["recommendations"].append("Practice more interview scenarios")
    else:
        feedback["performance"] = "Needs significant improvement"
        feedback["improvements"].extend([
            "Practice answering common HR questions",
            "Work on communication skills",
            "Build confidence through mock interviews"
        ])
        feedback["recommendations"].append("Consider interview coaching")
    
    return feedback

@hr_bp.route("/history", methods=["GET"])
def get_hr_history():
    """Get HR interview history"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT overall_score, clarity_score, confidence_score,
                   relevance_score, completed_at, company_name
            FROM hr_attempts
            WHERE student_email = ? AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 10
        """, (email,))
        
        history = []
        for row in c.fetchall():
            history.append({
                "overall_score": row[0],
                "clarity_score": row[1],
                "confidence_score": row[2],
                "relevance_score": row[3],
                "completed_at": row[4],
                "company": row[5]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "history": history
        })
    
    except Exception as e:
        print(f"Error getting HR history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Initialize HR tables
def init_hr_tables():
    """Initialize HR tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            total_questions INTEGER DEFAULT 10,
            clarity_score REAL DEFAULT 0,
            confidence_score REAL DEFAULT 0,
            relevance_score REAL DEFAULT 0,
            grammar_score REAL DEFAULT 0,
            overall_score REAL DEFAULT 0,
            total_time INTEGER,
            status TEXT DEFAULT 'in_progress',
            completed_at TIMESTAMP
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            answer TEXT NOT NULL,
            clarity_score REAL DEFAULT 0,
            confidence_score REAL DEFAULT 0,
            relevance_score REAL DEFAULT 0,
            grammar_score REAL DEFAULT 0,
            answer_score REAL DEFAULT 0,
            time_spent INTEGER,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(attempt_id) REFERENCES hr_attempts(id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ HR tables initialized!")

# Initialize on module load
init_hr_tables()