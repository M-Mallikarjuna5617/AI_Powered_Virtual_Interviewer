"""
Technical Interview Module
Handles coding questions, code execution, and technical assessments
"""

from flask import Blueprint, jsonify, request, session, render_template
import sqlite3
import json
import time
import random
from datetime import datetime
from ai_service import CodeExecutionService, execute_student_code

technical_bp = Blueprint("technical", __name__, url_prefix="/technical")

DB_PATH = "users.db"
code_executor = CodeExecutionService()

def init_technical_questions():
    """Initialize technical questions database with real company data"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS technical_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            domain TEXT NOT NULL,
            question TEXT NOT NULL,
            test_cases TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            year INTEGER NOT NULL,
            language TEXT NOT NULL
        )
    """)

    # Check if questions already exist
    c.execute("SELECT COUNT(*) FROM technical_questions")
    if c.fetchone()[0] > 0:
        print("✅ Technical questions already initialized!")
        conn.close()
        return

    # ✅ Merge both data sections into one clean dataset
    questions_data = [
        # TCS Questions
        (1, "Programming", "Write a function to reverse a string without using built-in functions.",
         '[{"input": "hello", "output": "olleh"}, {"input": "world", "output": "dlrow"}]',
         "easy", 2024, "python"),
        (1, "Programming", "Implement a function to find the factorial of a number using recursion.",
         '[{"input": "5", "output": "120"}, {"input": "3", "output": "6"}]',
         "medium", 2024, "python"),
        (1, "Data Structures", "Write a function to check if a string is a palindrome.",
         '[{"input": "racecar", "output": "True"}, {"input": "hello", "output": "False"}]',
         "easy", 2024, "python"),

        # Infosys
        (2, "Programming", "Implement a binary search algorithm.",
         '[{"input": "[1,2,3,4,5]\\n3", "output": "2"}, {"input": "[1,2,3,4,5]\\n6", "output": "-1"}]',
         "medium", 2024, "python"),
        (2, "Algorithms", "Write a function to find the maximum element in an array.",
         '[{"input": "[3,7,2,9,1]", "output": "9"}, {"input": "[-1,-5,-3]", "output": "-1"}]',
         "easy", 2024, "python"),

        # Wipro
        (3, "Programming", "Implement a stack using arrays with push, pop, and peek operations.",
         '[{"input": "push(5)\\npush(3)\\npeek()\\npop()", "output": "3\\n3"}]',
         "medium", 2024, "python"),
        (3, "Data Structures", "Write a function to remove duplicates from a list.",
         '[{"input": "[1,2,2,3,4,4,5]", "output": "[1,2,3,4,5]"}]',
         "easy", 2024, "python"),

        # Tech Mahindra
        (4, "Programming", "Implement a function to count the frequency of each character in a string.",
         '[{"input": "hello", "output": "{\'h\':1,\'e\':1,\'l\':2,\'o\':1}"}]',
         "medium", 2024, "python"),
        (4, "Algorithms", "Write a function to find the second largest number in an array.",
         '[{"input": "[3,7,2,9,1]", "output": "7"}, {"input": "[1,1,1]", "output": "None"}]',
         "medium", 2024, "python"),

        # Accenture
        (5, "Programming", "Implement a function to check if two strings are anagrams.",
         '[{"input": "listen\\nsilent", "output": "True"}, {"input": "hello\\nworld", "output": "False"}]',
         "medium", 2024, "python"),
        (5, "Data Structures", "Write a function to find the longest common subsequence of two strings.",
         '[{"input": "ABCDGH\\nAEDFHR", "output": "ADH"}]',
         "hard", 2024, "python"),

        # Cognizant
        (6, "Programming", "Implement a function to sort an array using bubble sort.",
         '[{"input": "[64,34,25,12,22,11,90]", "output": "[11,12,22,25,34,64,90]"}]',
         "medium", 2024, "python"),
        (6, "Algorithms", "Write a function to find the GCD of two numbers using Euclidean algorithm.",
         '[{"input": "48\\n18", "output": "6"}, {"input": "17\\n13", "output": "1"}]',
         "medium", 2024, "python"),

        # Capgemini
        (7, "Programming", "Implement a function to check if a number is prime.",
         '[{"input": "17", "output": "True"}, {"input": "15", "output": "False"}]',
         "easy", 2024, "python"),
        (7, "Data Structures", "Write a function to implement a queue using two stacks.",
         '[{"input": "enqueue(1)\\nenqueue(2)\\ndequeue()\\ndequeue()", "output": "1\\n2"}]',
         "hard", 2024, "python"),

        # L&T
        (8, "Programming", "Implement a function to find the nth Fibonacci number.",
         '[{"input": "5", "output": "5"}, {"input": "10", "output": "55"}]',
         "medium", 2024, "python"),
        (8, "Algorithms", "Write a function to implement merge sort.",
         '[{"input": "[38,27,43,3,9,82,10]", "output": "[3,9,10,27,38,43,82]"}]',
         "hard", 2024, "python"),

        # Mindtree
        (9, "Programming", "Implement a function to find the longest palindromic substring.",
         '[{"input": "babad", "output": "bab"}, {"input": "cbbd", "output": "bb"}]',
         "hard", 2024, "python"),
        (9, "Data Structures", "Write a function to implement a binary tree and find its height.",
         '[{"input": "1,2,3,4,5", "output": "3"}]',
         "medium", 2024, "python"),

        # HCL
        (10, "Programming", "Implement a function to find the intersection of two arrays.",
         '[{"input": "[1,2,2,1]\\n[2,2]", "output": "[2]"}, {"input": "[4,9,5]\\n[9,4,9,8,4]", "output": "[4,9]"}]',
         "medium", 2024, "python"),
        (10, "Algorithms", "Write a function to implement quick sort.",
         '[{"input": "[10,7,8,9,1,5]", "output": "[1,5,7,8,9,10]"}]',
         "hard", 2024, "python")
    ]

    c.executemany("""
        INSERT INTO technical_questions (company_id, domain, question, test_cases, difficulty, year, language)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, questions_data)
    
    conn.commit()
    conn.close()


@technical_bp.route("/")
def technical_home():
    """Technical interview home page"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    return render_template("technical.html", username=session.get("name"))

@technical_bp.route("/start", methods=["POST"])
def start_technical_interview():
    """Start a technical interview session"""
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
        
        # Get 3 random questions for the interview
        c.execute("""
            SELECT id, domain, question, test_cases, difficulty, language
            FROM technical_questions
            WHERE company_id = ? OR company_id = 1
            ORDER BY RANDOM()
            LIMIT 3
        """, (company_id,))
        
        questions = []
        for row in c.fetchall():
            questions.append({
                "id": row[0],
                "domain": row[1],
                "question": row[2],
                "test_cases": json.loads(row[3]),
                "difficulty": row[4],
                "language": row[5]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "company": company_name,
            "questions": questions,
            "time_limit": 45  # 45 minutes
        })
    
    except Exception as e:
        print(f"Error starting technical interview: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@technical_bp.route("/execute", methods=["POST"])
def execute_code():
    """Execute student's code against test cases"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        code = data.get('code', '')
        language = data.get('language', 'python')
        question_id = data.get('question_id')
        
        if not code.strip():
            return jsonify({"success": False, "error": "No code provided"}), 400
        
        # Get test cases for the question
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT test_cases FROM technical_questions WHERE id = ?", (question_id,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            return jsonify({"success": False, "error": "Question not found"}), 404
        
        test_cases = json.loads(result[0])
        
        # Execute code using AI service
        start_time = time.time()
        execution_result = code_executor.execute_code(code, language, test_cases)
        execution_time = time.time() - start_time
        
        # Save result to database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO technical_results 
            (student_email, company_name, question_id, code_submitted, language, 
             test_results, score, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session["email"], "Technical Interview", question_id, code, language,
              json.dumps(execution_result), execution_result['score'], execution_time))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "result": execution_result,
            "execution_time": execution_time
        })
    
    except Exception as e:
        print(f"Error executing code: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@technical_bp.route("/submit", methods=["POST"])
def submit_technical_interview():
    """Submit complete technical interview"""
    if "email" not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.json
        email = session["email"]
        solutions = data.get('solutions', [])  # List of {question_id, code, test_passed}
        
        # Calculate overall score
        total_questions = len(solutions)
        total_passed = sum(1 for sol in solutions if sol.get('test_passed', False))
        
        overall_score = (total_passed / total_questions * 100) if total_questions > 0 else 0
        
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
        
        # Save each solution
        for solution in solutions:
            c.execute("""
                INSERT INTO technical_results 
                (student_email, company_name, question_id, code_submitted, language, 
                 test_results, score, execution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                email, company_name, solution.get('question_id'),
                solution.get('code', ''), solution.get('language', 'python'),
                json.dumps(solution.get('test_results', [])),
                solution.get('score', 0), solution.get('execution_time', 0)
            ))
        
        conn.commit()
        conn.close()
        
        # Generate feedback
        feedback = generate_technical_feedback(solutions, overall_score)
        
        return jsonify({
            "success": True,
            "overall_score": round(overall_score, 2),
            "total_questions": total_questions,
            "passed_tests": total_passed,
            "feedback": feedback,
            "company": company_name
        })
    
    except Exception as e:
        print(f"Error submitting technical interview: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def generate_technical_feedback(solutions, overall_score):
    """Generate personalized feedback based on technical performance"""
    feedback = {
        "overall_score": overall_score,
        "strengths": [],
        "improvements": [],
        "recommendations": []
    }
    
    if overall_score >= 80:
        feedback["strengths"].append("Excellent problem-solving skills")
        feedback["strengths"].append("Strong coding fundamentals")
        feedback["strengths"].append("All test cases passed")
        feedback["recommendations"].append("Ready for senior technical roles")
        feedback["recommendations"].append("Consider practicing advanced algorithms")
    elif overall_score >= 60:
        feedback["strengths"].append("Good understanding of algorithms")
        feedback["improvements"].append("Practice more complex data structures")
        feedback["recommendations"].append("Focus on time complexity optimization")
        feedback["recommendations"].append("Review edge cases")
    else:
        feedback["improvements"].append("Strengthen basic programming concepts")
        feedback["improvements"].append("Practice more coding problems")
        feedback["recommendations"].append("Consider taking programming courses")
        feedback["recommendations"].append("Start with easy problems on LeetCode")
    
    # Analyze code quality
    total_tests = sum(len(s.get('test_results', [])) for s in solutions)
    passed_tests = sum(s.get('test_passed', 0) for s in solutions)
    
    if total_tests > 0:
        test_pass_rate = (passed_tests / total_tests * 100)
        if test_pass_rate >= 90:
            feedback["strengths"].append("High test case pass rate")
        elif test_pass_rate < 50:
            feedback["improvements"].append("Focus on passing all test cases")
    
    return feedback

# Initialize questions when module is imported
init_technical_questions()
