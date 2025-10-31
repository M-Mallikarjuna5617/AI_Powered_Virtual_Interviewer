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

# ✅ FIX 1: Use correct database paths
USERS_DB_PATH = "users.db"  # For storing test attempts and results
QUESTIONS_DB_PATH = "questions.db"  # For fetching questions


def init_aptitude_tables():
    """Initialize database tables for aptitude tests"""
    conn = sqlite3.connect(USERS_DB_PATH)
    c = conn.cursor()
    
    # ✅ Aptitude attempts table (stores in users.db)
    c.execute("""
        CREATE TABLE IF NOT EXISTS aptitude_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            total_questions INTEGER DEFAULT 30,
            correct_answers INTEGER DEFAULT 0,
            score REAL DEFAULT 0,
            time_taken INTEGER,
            status TEXT DEFAULT 'completed',
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_email) REFERENCES students(email)
        )
    """)
    
    # ✅ Aptitude responses table (stores in users.db)
    c.execute("""
        CREATE TABLE IF NOT EXISTS aptitude_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            question_id INTEGER,
            question_text TEXT,
            selected_answer TEXT,
            correct_answer TEXT,
            is_correct BOOLEAN,
            time_spent INTEGER,
            FOREIGN KEY(attempt_id) REFERENCES aptitude_attempts(id)
        )
    """)
    
    conn.commit()
    conn.close()
    
    # ✅ Initialize question bank in questions.db
    initialize_question_bank()
    print("✅ Aptitude test tables initialized!")


@aptitude_bp.route("/get-questions/<topic>", methods=["GET"])
def get_test_questions(topic):
    """Get 30 questions for the test"""
    print(f"Debug: Session data: {dict(session)}")
    if "email" not in session:
        print("Debug: No email in session")
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        # Get selected company
        selected_company = get_selected_company(email)
        company_name = selected_company['name'] if selected_company else "TCS"
        
        print(f"✅ Fetching questions for company: {company_name}")
        
        # ✅ FIX 2: Connect to questions.db (not users.db)
        conn = sqlite3.connect(QUESTIONS_DB_PATH)
        c = conn.cursor()
        
        # ✅ FIX 3: Get company_id first with normalized match
        c.execute("SELECT id FROM companies WHERE lower(trim(name)) = lower(trim(?))", (company_name,))
        company_result = c.fetchone()
        
        if not company_result:
            conn.close()
            return jsonify({
                "success": False,
                "error": f"Selected company '{company_name}' not found in questions database"
            }), 400
        else:
            company_id = company_result[0]
            # ✅ FIX 4: Query from aptitude_questions with company filter
            c.execute("""
                SELECT id, question, option_a, option_b, option_c, option_d, 
                       correct_answer, explanation, difficulty, category, time_limit
                FROM aptitude_questions 
                WHERE company_id = ?
                ORDER BY RANDOM()
                LIMIT 30
            """, (company_id,))
        
        rows = c.fetchall()
        conn.close()
        
        print(f"✅ Found {len(rows)} questions in database")
        
        # ✅ FIX 5: Build questions array; will top-up to 30 if fewer available
        questions = []
        for row in rows:
            questions.append({
                "id": row[0],
                "question": row[1],
                "options": [row[2], row[3], row[4], row[5]],
                "correct_answer": row[6],  # A, B, C, D
                "explanation": row[7],
                "difficulty": row[8] if row[8] else "Medium",
                "category": row[9] if row[9] else "General",
                "time_limit": row[10] if row[10] else 60
            })
        
        # ✅ Ensure exactly 30 questions by sampling within the same company if needed
        if len(questions) < 30:
            print(f"⚠️ Warning: Only {len(questions)} questions available for {company_name}. Topping up to 30 by sampling with replacement.")
            import random
            if questions:
                while len(questions) < 30:
                    q = random.choice(questions)
                    # Create a shallow copy to avoid shared references
                    questions.append({
                        "id": q["id"],
                        "question": q["question"],
                        "options": list(q["options"]),
                        "correct_answer": q["correct_answer"],
                        "explanation": q["explanation"],
                        "difficulty": q["difficulty"],
                        "category": q["category"],
                        "time_limit": q["time_limit"],
                    })
            # If there are zero questions, return empty result gracefully
        
        return jsonify({
            "success": True,
            "questions": questions,
            "total_questions": len(questions),
            "company": company_name
        })
    
    except Exception as e:
        print(f"❌ Error fetching questions: {e}")
        import traceback
        traceback.print_exc()
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
        
        print(f"✅ Completing test for {email} with {len(responses)} responses")
        
        # Get selected company
        selected_company = get_selected_company(email)
        company_name = selected_company['name'] if selected_company else "General"
        
        # ✅ FIX 6: Save to users.db
        users_conn = sqlite3.connect(USERS_DB_PATH)
        users_c = users_conn.cursor()
        
        # ✅ Create aptitude attempt
        users_c.execute("""
            INSERT INTO aptitude_attempts 
            (student_email, company_name, total_questions, time_taken, status)
            VALUES (?, ?, ?, ?, 'completed')
        """, (email, company_name, len(responses), time_taken))
        
        attempt_id = users_c.lastrowid
        print(f"✅ Created attempt_id: {attempt_id}")
        
        # ✅ FIX 7: Connect to questions.db to get correct answers
        questions_conn = sqlite3.connect(QUESTIONS_DB_PATH)
        questions_c = questions_conn.cursor()
        
        # Calculate score and save each response
        correct_count = 0
        
        answered_with_keys = 0
        for resp in responses:
            question_id = resp.get('question_id')
            selected = resp.get('selected_answer')  # A, B, C, D
            time_spent = resp.get('time_spent', 0)
            
            # ✅ Get correct answer from questions.db
            questions_c.execute("""
                SELECT correct_answer, question 
                FROM aptitude_questions 
                WHERE id = ?
            """, (question_id,))
            
            result = questions_c.fetchone()
            if result:
                correct_answer = result[0]  # A, B, C, D or None
                question_text = result[1]
                is_correct = None
                if correct_answer in ("A", "B", "C", "D"):
                    is_correct = (selected == correct_answer)
                    answered_with_keys += 1
                    if is_correct:
                        correct_count += 1
                
                # ✅ Save response to users.db
                users_c.execute("""
                    INSERT INTO aptitude_responses 
                    (attempt_id, question_id, question_text, selected_answer, 
                     correct_answer, is_correct, time_spent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (attempt_id, question_id, question_text, selected, 
                      correct_answer, is_correct, time_spent))
        
        questions_conn.close()
        
        # ✅ Calculate score
        # Score using only questions that have correct answers
        score = (correct_count / answered_with_keys * 100) if answered_with_keys else 0
        
        # ✅ Update attempt with final score
        users_c.execute("""
            UPDATE aptitude_attempts 
            SET correct_answers = ?, score = ?
            WHERE id = ?
        """, (correct_count, score, attempt_id))
        
        users_conn.commit()
        users_conn.close()
        
        print(f"✅ Test completed: Score={score:.2f}%, Correct={correct_count}/{len(responses)}")
        
        return jsonify({
            "success": True,
            "message": "Test completed successfully",
            "score": round(score, 2),
            "correct_answers": correct_count,
            "total_questions": len(responses),
            "attempt_id": attempt_id
        })

    except Exception as e:
        print(f"❌ Error completing test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@aptitude_bp.route("/history", methods=["GET"])
def get_aptitude_history():
    """Get student's aptitude test history"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(USERS_DB_PATH)
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
        
        conn = sqlite3.connect(USERS_DB_PATH)
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
            conn.close()
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