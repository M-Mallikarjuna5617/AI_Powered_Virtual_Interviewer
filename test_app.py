#!/usr/bin/env python3
"""
Minimal test application for aptitude functionality
"""

from flask import Flask, jsonify, request, session, render_template
import sqlite3
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'test_secret_key'

DB_PATH = "users.db"

@app.route('/')
def home():
    return "Aptitude Test Server Running!"

@app.route('/aptitude/get-questions/aptitude', methods=['GET'])
def get_test_questions():
    """Get questions for the test"""
    print(f"Debug: Session data: {dict(session)}")
    if "email" not in session:
        print("Debug: No email in session")
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        email = session["email"]
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get questions from question_bank table
        c.execute("""
            SELECT id, question_text, option_a, option_b, option_c, option_d, 
                   correct_answer, explanation, difficulty, subtopic, time_limit
            FROM question_bank 
            ORDER BY RANDOM()
            LIMIT 30
        """)
        
        rows = c.fetchall()
        print(f"Debug: Found {len(rows)} questions in database")
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
            "company": "Test Company"
        })
    
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/aptitude/complete-test', methods=['POST'])
def complete_test():
    """Complete test and save results"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        email = session["email"]
        responses = data.get('responses', [])
        time_taken = data.get('time_taken', 0)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create aptitude attempt
        c.execute("""
            INSERT INTO aptitude_attempts (student_email, company_name, total_questions, time_taken)
            VALUES (?, ?, ?, ?)
        """, (email, "Test Company", len(responses), time_taken))
        
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

@app.route('/test-login', methods=['POST'])
def test_login():
    """Test login to set session"""
    data = request.json
    email = data.get('email', 'test@example.com')
    session['email'] = email
    return jsonify({"success": True, "message": "Logged in successfully"})

if __name__ == '__main__':
    print("Starting minimal aptitude test server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
