# ============================================
# FILE: enhanced_question_generator.py
# Complete Question Bank with 200+ Questions
# Full Detailed Version (No Minimization)
# ============================================

import sqlite3
import json
from datetime import datetime
import random

DB_PATH = "questions.db"  # Unified DB name for all modules

# =======================
# CREATE TABLES
# =======================
def create_tables_if_not_exist():
    """Create all tables if they do not exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Aptitude
    c.execute("""
        CREATE TABLE IF NOT EXISTS aptitude_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            category TEXT,
            difficulty TEXT,
            question TEXT,
            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,
            correct_answer TEXT,
            explanation TEXT,
            time_limit INTEGER,
            year_asked INTEGER
        )
    """)

    # Technical
    c.execute("""
        CREATE TABLE IF NOT EXISTS technical_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            difficulty TEXT,
            question_title TEXT,
            question_description TEXT,
            input_format TEXT,
            output_format TEXT,
            constraints TEXT,
            sample_input TEXT,
            sample_output TEXT,
            test_cases TEXT,
            time_limit INTEGER,
            year_asked INTEGER,
            tags TEXT
        )
    """)

    # GD
    c.execute("""
        CREATE TABLE IF NOT EXISTS gd_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            topic TEXT,
            category TEXT,
            description TEXT,
            key_points TEXT,
            evaluation_criteria TEXT,
            time_limit INTEGER,
            year_asked INTEGER
        )
    """)

    # HR
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            question TEXT,
            category TEXT,
            sample_answer TEXT,
            evaluation_points TEXT,
            difficulty TEXT,
            year_asked INTEGER
        )
    """)

    # Companies
    c.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Tables checked/created successfully!")


# =======================
# INITIALIZE QUESTION BANK
# =======================
def initialize_question_bank():
    """Initialize the complete question bank for all companies."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Ensure tables exist
    create_tables_if_not_exist()

    # Seed companies if not present
    companies_list = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Amazon", "Google"]
    for company in companies_list:
        c.execute("INSERT OR IGNORE INTO companies (name) VALUES (?)", (company,))
    conn.commit()

    # Fetch company IDs
    c.execute("SELECT id, name FROM companies")
    companies = {name: cid for cid, name in c.fetchall()}

    print("📝 Populating comprehensive question banks...")

    # =======================
    # APTITUDE QUESTIONS
    # =======================
    aptitude_questions = []

    # Logical Reasoning
    logical_questions = [
        {
            "category": "Logical Reasoning",
            "difficulty": "Easy",
            "question": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
            "options": ["Yes", "No", "Cannot be determined", "Insufficient data"],
            "correct": "A",
            "explanation": "Valid syllogism: A⊆B and B⊆C implies A⊆C"
        },
        {
            "category": "Logical Reasoning",
            "difficulty": "Medium",
            "question": "In a certain code, GRAPE is 27354 and FOUR is 1687. What is GROUP?",
            "options": ["27685", "27865", "27684", "27864"],
            "correct": "D",
            "explanation": "G=2, R=7, O=8, U=6, P=4"
        },
        {
            "category": "Logical Reasoning",
            "difficulty": "Hard",
            "question": "If South-East becomes North and North becomes South-West, what does West become?",
            "options": ["North-East", "North-West", "South-East", "South"],
            "correct": "A",
            "explanation": "135° clockwise rotation"
        },
        {
            "category": "Logical Reasoning",
            "difficulty": "Easy",
            "question": "Complete the series: 2, 6, 12, 20, 30, ?",
            "options": ["40", "42", "44", "46"],
            "correct": "B",
            "explanation": "Add 4, 6, 8, 10, 12"
        },
        {
            "category": "Logical Reasoning",
            "difficulty": "Medium",
            "question": "Find the odd one: Dog, Cat, Lion, Snake",
            "options": ["Dog", "Cat", "Lion", "Snake"],
            "correct": "D",
            "explanation": "Snake is reptile, others are mammals"
        }
    ]

    # Quantitative Aptitude
    quant_questions = [
        {
            "category": "Quantitative Aptitude",
            "difficulty": "Easy",
            "question": "What is 15% of 200?",
            "options": ["25", "30", "35", "40"],
            "correct": "B",
            "explanation": "15/100 × 200 = 30"
        },
        {
            "category": "Quantitative Aptitude",
            "difficulty": "Medium",
            "question": "A train travels 60 km in 45 minutes. Speed in km/hr?",
            "options": ["70", "75", "80", "85"],
            "correct": "C",
            "explanation": "60/(45/60) = 80 km/hr"
        },
        {
            "category": "Quantitative Aptitude",
            "difficulty": "Hard",
            "question": "If P:Q = 2:3 and Q:R = 4:5, find P:R",
            "options": ["8:15", "2:5", "3:5", "4:15"],
            "correct": "A",
            "explanation": "P:Q:R = 8:12:15, so P:R = 8:15"
        },
        {
            "category": "Quantitative Aptitude",
            "difficulty": "Easy",
            "question": "Simple Interest on Rs.1000 at 10% for 2 years?",
            "options": ["100", "150", "200", "250"],
            "correct": "C",
            "explanation": "SI = PRT/100 = 1000×10×2/100 = 200"
        },
        {
            "category": "Quantitative Aptitude",
            "difficulty": "Medium",
            "question": "Average of 5, 10, 15, 20, 25?",
            "options": ["12", "15", "18", "20"],
            "correct": "B",
            "explanation": "Sum = 75, Avg = 75/5 = 15"
        }
    ]

    # Verbal Ability
    verbal_questions = [
        {
            "category": "Verbal Ability",
            "difficulty": "Easy",
            "question": "Synonym of AMBIGUOUS:",
            "options": ["Clear", "Vague", "Precise", "Definite"],
            "correct": "B",
            "explanation": "Ambiguous means unclear"
        },
        {
            "category": "Verbal Ability",
            "difficulty": "Medium",
            "question": "Antonym of ABUNDANT:",
            "options": ["Scarce", "Plenty", "Rich", "Ample"],
            "correct": "A",
            "explanation": "Opposite of abundant is scarce"
        },
        {
            "category": "Verbal Ability",
            "difficulty": "Easy",
            "question": "Fill: He is allergic ___ dust",
            "options": ["at", "to", "from", "with"],
            "correct": "B",
            "explanation": "Allergic to is correct usage"
        },
        {
            "category": "Verbal Ability",
            "difficulty": "Medium",
            "question": "Spot error: Neither he nor I am going",
            "options": ["Neither", "nor", "am", "No error"],
            "correct": "D",
            "explanation": "Sentence is correct"
        },
        {
            "category": "Verbal Ability",
            "difficulty": "Hard",
            "question": "Sentence improvement: I have been living here since five years",
            "options": ["for five years", "from five years", "in five years", "No improvement"],
            "correct": "A",
            "explanation": "Use 'for' with duration"
        }
    ]

    # Data Interpretation
    data_questions = [
        {
            "category": "Data Interpretation",
            "difficulty": "Medium",
            "question": "Profit increased from 20L to 25L. Percentage increase?",
            "options": ["20%", "25%", "30%", "15%"],
            "correct": "B",
            "explanation": "(5/20)×100 = 25%"
        },
        {
            "category": "Data Interpretation",
            "difficulty": "Easy",
            "question": "If 40% of students passed, what % failed?",
            "options": ["40%", "50%", "60%", "70%"],
            "correct": "C",
            "explanation": "100 - 40 = 60%"
        },
        {
            "category": "Data Interpretation",
            "difficulty": "Hard",
            "question": "Sales increased by 20% then decreased by 20%. Net change?",
            "options": ["0%", "-4%", "+4%", "-2%"],
            "correct": "B",
            "explanation": "1.2×0.8 = 0.96, so 4% decrease"
        }
    ]

    all_aptitude = logical_questions + quant_questions + verbal_questions + data_questions

    # Insert aptitude questions per company
    for cname, cid in companies.items():
        for q in all_aptitude:
            aptitude_questions.append({
                "company_id": cid,
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "option_a": q["options"][0],
                "option_b": q["options"][1],
                "option_c": q["options"][2],
                "option_d": q["options"][3],
                "correct_answer": q["correct"],
                "explanation": q["explanation"],
                "time_limit": 60,
                "year_asked": 2023 + (hash(cname) % 2)
            })

    c.execute("DELETE FROM aptitude_questions")
    for q in aptitude_questions:
        c.execute("""
            INSERT INTO aptitude_questions (
                company_id, category, difficulty, question,
                option_a, option_b, option_c, option_d,
                correct_answer, explanation, time_limit, year_asked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(q.values()))

    print(f"✅ Added {len(aptitude_questions)} aptitude questions")

    # =======================
    # TECHNICAL QUESTIONS
    # =======================
    technical_questions = [
        {
            "difficulty": "Easy",
            "title": "Two Sum",
            "description": "Find two numbers that add to target",
            "input_format": "Array and target",
            "output_format": "Indices",
            "constraints": "2 ≤ n ≤ 10^4",
            "sample_input": "[2,7,11,15], 9",
            "sample_output": "[0,1]",
            "test_cases": json.dumps([
                {"input": "[2,7,11,15], 9", "output": "[0,1]"},
                {"input": "[3,2,4], 6", "output": "[1,2]"}
            ]),
            "tags": "Array,Hash Table"
        },
        {
            "difficulty": "Easy",
            "title": "Reverse String",
            "description": "Reverse a given string",
            "input_format": "String s",
            "output_format": "Reversed string",
            "constraints": "1 ≤ len ≤ 1000",
            "sample_input": "hello",
            "sample_output": "olleh",
            "test_cases": json.dumps([
                {"input": "hello", "output": "olleh"},
                {"input": "python", "output": "nohtyp"}
            ]),
            "tags": "String"
        },
        {
            "difficulty": "Medium",
            "title": "Valid Parentheses",
            "description": "Check if parentheses are balanced",
            "input_format": "String with brackets",
            "output_format": "True/False",
            "constraints": "1 ≤ len ≤ 10^4",
            "sample_input": "()[]{}",
            "sample_output": "True",
            "test_cases": json.dumps([
                {"input": "()[]{}", "output": "True"},
                {"input": "(]", "output": "False"}
            ]),
            "tags": "Stack,String"
        },
        {
            "difficulty": "Medium",
            "title": "Merge Sorted Arrays",
            "description": "Merge two sorted arrays",
            "input_format": "Two sorted arrays",
            "output_format": "Merged sorted array",
            "constraints": "0 ≤ m,n ≤ 200",
            "sample_input": "[1,2,3], [2,5,6]",
            "sample_output": "[1,2,2,3,5,6]",
            "test_cases": json.dumps([
                {"input": "[1,2,3], [2,5,6]", "output": "[1,2,2,3,5,6]"}
            ]),
            "tags": "Array,Two Pointers"
        },
        {
            "difficulty": "Hard",
            "title": "Longest Substring Without Repeating",
            "description": "Find longest substring without repeating characters",
            "input_format": "String s",
            "output_format": "Integer length",
            "constraints": "0 ≤ len ≤ 5×10^4",
            "sample_input": "abcabcbb",
            "sample_output": "3",
            "test_cases": json.dumps([
                {"input": "abcabcbb", "output": "3"},
                {"input": "bbbbb", "output": "1"}
            ]),
            "tags": "String,Sliding Window"
        }
    ]

    c.execute("DELETE FROM technical_questions")
    for cname, cid in companies.items():
        for q in technical_questions:
            c.execute("""
                INSERT INTO technical_questions (
                    company_id, difficulty, question_title, question_description,
                    input_format, output_format, constraints, sample_input,
                    sample_output, test_cases, time_limit, year_asked, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1800, 2023, ?)
            """, (cid, q["difficulty"], q["title"], q["description"],
                  q["input_format"], q["output_format"], q["constraints"],
                  q["sample_input"], q["sample_output"], q["test_cases"], q["tags"]))

    print(f"✅ Added {len(technical_questions) * len(companies)} technical questions")

    # =======================
    # GD TOPICS
    # =======================
    gd_topics = [
        {
            "topic": "AI: Boon or Bane for Employment",
            "category": "Technology",
            "description": "Impact of AI on jobs",
            "key_points": json.dumps(["Automation", "Job creation", "Reskilling", "Ethics"])
        },
        {
            "topic": "Remote Work vs Office Work",
            "category": "Workplace",
            "description": "Future of work culture",
            "key_points": json.dumps(["Flexibility", "Productivity", "Team collaboration", "Work-life balance"])
        },
        {
            "topic": "Social Media Impact on Youth",
            "category": "Social",
            "description": "Effects on mental health and productivity",
            "key_points": json.dumps(["Mental health", "Productivity", "Connectivity", "Privacy"])
        },
        {
            "topic": "Electric Vehicles Future in India",
            "category": "Environment",
            "description": "Sustainability and infrastructure",
            "key_points": json.dumps(["Environment", "Infrastructure", "Cost", "Technology"])
        },
        {
            "topic": "Online Education vs Traditional",
            "category": "Education",
            "description": "Learning effectiveness comparison",
            "key_points": json.dumps(["Accessibility", "Quality", "Interaction", "Cost"])
        }
    ]

    c.execute("DELETE FROM gd_topics")
    for cname, cid in companies.items():
        for topic in gd_topics:
            c.execute("""
                INSERT INTO gd_topics (
                    company_id, topic, category, description,
                    key_points, evaluation_criteria, time_limit, year_asked
                ) VALUES (?, ?, ?, ?, ?, 'Content,Communication,Leadership,Teamwork', 600, 2024)
            """, (cid, topic["topic"], topic["category"], topic["description"], topic["key_points"]))

    print(f"✅ Added {len(gd_topics) * len(companies)} GD topics")

    # =======================
    # HR QUESTIONS
    # =======================
    hr_questions = [
        {
            "question": "Tell me about yourself",
            "category": "Introduction",
            "difficulty": "Easy",
            "sample_answer": "Start with education, key projects, skills, career goals"
        },
        {
            "question": "Why do you want to join our company?",
            "category": "Company Knowledge",
            "difficulty": "Medium",
            "sample_answer": "Research company values, culture, growth opportunities"
        },
        {
            "question": "What are your strengths?",
            "category": "Self-Assessment",
            "difficulty": "Easy",
            "sample_answer": "Specific skills with examples"
        },
        {
            "question": "What are your weaknesses?",
            "category": "Self-Assessment",
            "difficulty": "Medium",
            "sample_answer": "Honest weakness with improvement plan"
        },
        {
            "question": "Describe a challenging situation you faced",
            "category": "Behavioral",
            "difficulty": "Hard",
            "sample_answer": "Use STAR method: Situation, Task, Action, Result"
        }
    ]

    c.execute("DELETE FROM hr_questions")
    for cname, cid in companies.items():
        for q in hr_questions:
            c.execute("""
                INSERT INTO hr_questions (
                    company_id, question, category, sample_answer, evaluation_points,
                    difficulty, year_asked
                ) VALUES (?, ?, ?, ?, 'Clarity,Confidence,Relevance', ?, 2024)
            """, (cid, q["question"], q["category"], q["sample_answer"], q["difficulty"]))

    print(f"✅ Added {len(hr_questions) * len(companies)} HR questions")

    conn.commit()
    conn.close()
    print(f"🎉 Question bank fully populated for {len(companies)} companies!")
# ===============================
# FUNCTION: Fetch Questions per Company
# ===============================
def get_questions_for_company(company_name, module="aptitude", limit=5):
    """
    Returns a list of questions for the specified company and module.
    module: 'aptitude', 'technical', 'gd', 'hr'
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get company ID
    c.execute("SELECT id FROM companies WHERE name=?", (company_name,))
    result = c.fetchone()
    if not result:
        conn.close()
        return []
    company_id = result[0]

    table_map = {
        "aptitude": "aptitude_questions",
        "technical": "technical_questions",
        "gd": "gd_topics",
        "hr": "hr_questions"
    }

    if module not in table_map:
        conn.close()
        return []

    table_name = table_map[module]

    # Fetch random questions
    c.execute(f"SELECT * FROM {table_name} WHERE company_id=? ORDER BY RANDOM() LIMIT ?", (company_id, limit))
    questions = c.fetchall()
    columns = [col[0] for col in c.description]

    conn.close()
    # Return as list of dicts
    return [dict(zip(columns, q)) for q in questions]
def get_adaptive_questions(user_score: int, limit: int = 10):
    """
    Dynamically fetch aptitude questions based on user_score.
    Higher score → harder questions.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if user_score >= 80:
        difficulty = "Hard"
    elif user_score >= 50:
        difficulty = "Medium"
    else:
        difficulty = "Easy"

    c.execute(
        "SELECT * FROM aptitude_questions WHERE difficulty=? ORDER BY RANDOM() LIMIT ?",
        (difficulty, limit)
    )
    questions = c.fetchall()
    columns = [col[0] for col in c.description]

    conn.close()
    return [dict(zip(columns, q)) for q in questions]


if __name__ == "__main__":
    create_tables_if_not_exist()
    initialize_question_bank()
