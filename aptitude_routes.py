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
        
        # Get student's performance history for adaptive questions
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            SELECT topic, subtopic, accuracy 
            FROM student_analytics 
            WHERE student_email = ?
        """, (email,))
        
        performance_history = [
            {"topic": row[0], "subtopic": row[1], "accuracy": row[2]}
            for row in c.fetchall()
        ]
        conn.close()
        
        # Get adaptive or company-specific questions
        if performance_history and len(performance_history) > 5:
            questions = get_adaptive_questions(email, company_name, performance_history)
        else:
            questions = get_questions_for_company(company_name, num_questions=20)
        
        # Remove correct answers from response
        for q in questions:
            q.pop('correct_answer', None)
            q.pop('explanation', None)
        
        return jsonify({
            "success": True,
            "questions": questions,
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
    """Complete test and calculate results"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        attempt_id = data.get('attempt_id')
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Calculate results
        c.execute("""
            SELECT COUNT(*), SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END)
            FROM test_responses
            WHERE attempt_id = ?
        """, (attempt_id,))
        
        total, correct = c.fetchone()
        score = (correct / total * 100) if total > 0 else 0
        
        # Update attempt
        c.execute("""
            UPDATE test_attempts 
            SET end_time = CURRENT_TIMESTAMP,
                correct_answers = ?,
                score = ?,
                status = 'completed'
            WHERE id = ?
        """, (correct, score, attempt_id))
        
        # Get subtopic-wise performance
        c.execute("""
            SELECT subtopic, difficulty,
                   COUNT(*) as total,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM test_responses
            WHERE attempt_id = ?
            GROUP BY subtopic, difficulty
        """, (attempt_id,))
        
        subtopic_performance = []
        for row in c.fetchall():
            subtopic, difficulty, total, correct = row
            accuracy = (correct / total * 100) if total > 0 else 0
            subtopic_performance.append({
                "subtopic": subtopic,
                "difficulty": difficulty,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "needs_improvement": accuracy < 60
            })
            
                        # Update analytics
            c.execute("""
                INSERT INTO student_analytics (student_email, topic, subtopic, total_attempted, correct_count, accuracy, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(student_email, topic, subtopic)
                DO UPDATE SET
                    total_attempted = total_attempted + excluded.total_attempted,
                    correct_count = correct_count + excluded.correct_count,
                    accuracy = ROUND(
                        CAST(correct_count + excluded.correct_count AS REAL) /
                        CAST(total_attempted + excluded.total_attempted AS REAL) * 100, 2
                    ),
                    last_updated = CURRENT_TIMESTAMP
            """, (email, "Aptitude", subtopic, total, correct, accuracy))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Test completed successfully",
            "score": score,
            "correct_answers": correct,
            "total_questions": total,
            "subtopic_performance": subtopic_performance
        })

    except Exception as e:
        print(f"Error completing test: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
