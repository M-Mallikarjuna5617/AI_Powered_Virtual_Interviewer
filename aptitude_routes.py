"""
Backend API routes for Aptitude Test System
Handles test attempts, question serving, and result calculation
"""

from flask import Blueprint, jsonify, request, session, redirect, url_for, flash
import sqlite3
from datetime import datetime
import json

# Import question generator
from question_generator import (
    initialize_question_bank,
    get_questions_for_company,
    get_adaptive_questions
)
from utils import get_selected_company

aptitude_bp = Blueprint("aptitude", __name__, url_prefix="/aptitude")

DB_PATH = "users.db"


def init_aptitude_tables():
    """Initialize database tables for aptitude tests"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Test attempts table
    c.execute("""
        CREATE TABLE IF NOT EXISTS test_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            topic TEXT NOT NULL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            total_questions INTEGER DEFAULT 20,
            correct_answers INTEGER DEFAULT 0,
            score REAL DEFAULT 0,
            status TEXT DEFAULT 'in_progress',
            FOREIGN KEY(student_email) REFERENCES students(email)
        )
    """)
    
    # Individual question responses
    c.execute("""
        CREATE TABLE IF NOT EXISTS test_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            question_text TEXT,
            selected_answer TEXT,
            correct_answer TEXT,
            is_correct BOOLEAN DEFAULT 0,
            time_spent INTEGER DEFAULT 0,
            subtopic TEXT,
            difficulty TEXT,
            FOREIGN KEY(attempt_id) REFERENCES test_attempts(id)
        )
    """)
    
    # Performance analytics
    c.execute("""
        CREATE TABLE IF NOT EXISTS student_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            topic TEXT NOT NULL,
            subtopic TEXT NOT NULL,
            total_attempted INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            accuracy REAL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(student_email, topic, subtopic)
        )
    """)
    
    conn.commit()
    conn.close()
    
    # Initialize question bank
    initialize_question_bank()
    print("✅ Aptitude test tables initialized!")


@aptitude_bp.route("/start/<topic>", methods=["POST"])
def start_test(topic):
    """Start a new test attempt"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        # Get selected company
        selected_company = get_selected_company(email)
        company_name = selected_company['name'] if selected_company else None
        
        # Create test attempt
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO test_attempts (student_email, company_name, topic, total_questions)
            VALUES (?, ?, ?, 20)
        """, (email, company_name, topic))
        
        attempt_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "attempt_id": attempt_id,
            "company": company_name,
            "topic": topic
        })
    
    except Exception as e:
        print(f"Error starting test: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@aptitude_bp.route("/get-questions/<topic>", methods=["GET"])
def get_test_questions(topic):
    """Get questions for the test"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        # Get selected company
        selected_company = get_selected_company(email)
        company_name = selected_company['name'] if selected_company else "General"
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get questions from question_bank table
        c.execute("""
            SELECT id, question_text, option_a, option_b, option_c, option_d, 
                   correct_answer, explanation, difficulty, subtopic, time_limit
            FROM question_bank 
            LIMIT 30
        """)
        
        rows = c.fetchall()
        questions = []
        for row in rows:
            questions.append({
                "id": row[0],
                "question": row[1],
                "options": [row[2], row[3], row[4], row[5]],
                "correct_answer": row[6],
                "explanation": row[7],
                "difficulty": row[8],
                "subtopic": row[9],
                "time_limit": row[10]
            })
        
        conn.close()
        
        # Shuffle and select 28 questions
        import random
        if len(questions) > 28:
            questions = random.sample(questions, 28)
        
        # Prepare response (without correct answers)
        response_questions = []
        for q in questions:
            response_questions.append({
                "id": q["id"],
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],  # Keep for validation
                "explanation": q["explanation"],
                "difficulty": q["difficulty"],
                "time_limit": q["time_limit"]
            })
        
        return jsonify({
            "success": True,
            "questions": response_questions,
            "total_questions": len(response_questions),
            "company": company_name
        })
    
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@aptitude_bp.route("/submit-answer", methods=["POST"])
def submit_answer():
    """Submit answer for a question"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        attempt_id = data.get('attempt_id')
        question_id = data.get('question_id')
        answer = data.get('answer')
        time_spent = data.get('time_spent', 0)
        
        # Get question details and verify answer
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            SELECT question_text, correct_answer, difficulty, subtopic 
            FROM question_bank 
            WHERE id = ?
        """, (question_id,))
        
        question = c.fetchone()
        
        if not question:
            conn.close()
            return jsonify({"success": False, "error": "Question not found"}), 404
        
        question_text, correct_answer, difficulty, subtopic = question
        is_correct = (answer == correct_answer)
        
        # Save response
        c.execute("""
            INSERT INTO test_responses 
            (attempt_id, question_id, question_text, selected_answer, correct_answer, 
             is_correct, time_spent, subtopic, difficulty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (attempt_id, question_id, question_text, answer, correct_answer, 
              is_correct, time_spent, subtopic, difficulty))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "is_correct": is_correct
        })
    
    except Exception as e:
        print(f"Error submitting answer: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@aptitude_bp.route("/complete-test", methods=["POST"])
def complete_test():
    """Complete test and save results"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        email = session["email"]
        responses = data.get('responses', [])  # List of {question_id, selected_answer, time_spent}
        time_taken = data.get('time_taken', 0)
        
        # Get selected company
        selected_company = get_selected_company(email)
        company_name = selected_company['name'] if selected_company else "General"
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create aptitude attempt
        c.execute("""
            INSERT INTO aptitude_attempts (student_email, company_name, total_questions, time_taken)
            VALUES (?, ?, ?, ?)
        """, (email, company_name, len(responses), time_taken))
        
        attempt_id = c.lastrowid
        
        # Calculate score and save responses
        correct_count = 0
        for resp in responses:
            question_id = resp.get('question_id')
            selected = resp.get('selected_answer')
            time_spent = resp.get('time_spent', 0)
            
            # Get correct answer
            c.execute("""
                SELECT correct_answer, question_text FROM question_bank WHERE id = ?
            """, (question_id,))
            
            result = c.fetchone()
            if result:
                correct_answer = result[0]
                question_text = result[1]
                is_correct = (selected == correct_answer)
                
                if is_correct:
                    correct_count += 1
                
                # Save response
                c.execute("""
                    INSERT INTO aptitude_responses 
                    (attempt_id, question_id, question_text, selected_answer, correct_answer, is_correct, time_spent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (attempt_id, question_id, question_text, selected, correct_answer, is_correct, time_spent))
        
        # Calculate score
        score = (correct_count / len(responses) * 100) if responses else 0
        
        # Update attempt with score
        c.execute("""
            UPDATE aptitude_attempts 
            SET correct_answers = ?, score = ?
            WHERE id = ?
        """, (correct_count, score, attempt_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "Test completed successfully",
            "score": round(score, 2),
            "correct_answers": correct_count,
            "total_questions": len(responses),
            "attempt_id": attempt_id
        })

    except Exception as e:
        print(f"Error completing test: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@aptitude_bp.route("/history", methods=["GET"])
def get_aptitude_history():
    """Get student's aptitude test history"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            SELECT id, company_name, score, correct_answers, total_questions, 
                   time_taken, completed_at
            FROM aptitude_attempts
            WHERE student_email = ?
            ORDER BY completed_at DESC
            LIMIT 10
        """, (email,))
        
        history = []
        for row in c.fetchall():
            history.append({
                "attempt_id": row[0],
                "company": row[1],
                "score": row[2],
                "correct_answers": row[3],
                "total_questions": row[4],
                "time_taken": row[5],
                "completed_at": row[6]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "history": history
        })
    
    except Exception as e:
        print(f"Error getting aptitude history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@aptitude_bp.route("/attempt/<attempt_id>", methods=["GET"])
def get_attempt_details(attempt_id):
    """Get detailed results for an aptitude attempt"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get attempt info
        c.execute("""
            SELECT company_name, score, correct_answers, total_questions, 
                   time_taken, completed_at
            FROM aptitude_attempts
            WHERE id = ? AND student_email = ?
        """, (attempt_id, email))
        
        attempt = c.fetchone()
        if not attempt:
            return jsonify({"success": False, "error": "Attempt not found"}), 404
        
        # Get responses
        c.execute("""
            SELECT question_text, selected_answer, correct_answer, is_correct, time_spent
            FROM aptitude_responses
            WHERE attempt_id = ?
        """, (attempt_id,))
        
        responses = []
        for row in c.fetchall():
            responses.append({
                "question": row[0],
                "selected": row[1],
                "correct": row[2],
                "is_correct": row[3],
                "time_spent": row[4]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "attempt": {
                "company": attempt[0],
                "score": attempt[1],
                "correct_answers": attempt[2],
                "total_questions": attempt[3],
                "time_taken": attempt[4],
                "completed_at": attempt[5]
            },
            "responses": responses
        })
    
    except Exception as e:
        print(f"Error getting attempt details: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
