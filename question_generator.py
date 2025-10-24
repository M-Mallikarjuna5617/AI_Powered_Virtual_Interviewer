"""
Populate comprehensive question banks for all interview rounds
Based on actual questions from Karnataka campus placements (2013–2024)
"""

import sqlite3
import json

DB_PATH = "users.db"

def populate_aptitude_questions():
    """Add aptitude questions from past 10–12 years"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, name FROM companies")
    companies = {name: id for id, name in c.fetchall()}
    
    aptitude_questions = [
        {
            "company_id": companies.get("TCS"),
            "category": "Logical Reasoning",
            "difficulty": "Easy",
            "question": "If in a certain code, GRAPE is written as 27354 and FOUR is written as 1687, how is GROUP written in that code?",
            "option_a": "27685",
            "option_b": "27865",
            "option_c": "27684",
            "option_d": "27864",
            "correct_answer": "D",
            "explanation": "G=2, R=7, O=8, U=6, P=4 → GROUP = 27864",
            "time_limit": 60,
            "year_asked": 2023
        },
        {
            "company_id": companies.get("Infosys"),
            "category": "Number Series",
            "difficulty": "Medium",
            "question": "Find the next number in the series: 2, 6, 12, 20, 30, ?",
            "option_a": "40",
            "option_b": "42",
            "option_c": "44",
            "option_d": "46",
            "correct_answer": "B",
            "explanation": "Differences: +4, +6, +8, +10, +12 → Next = 30 + 12 = 42",
            "time_limit": 60,
            "year_asked": 2024
        },
        {
            "company_id": companies.get("Wipro"),
            "category": "Data Interpretation",
            "difficulty": "Medium",
            "question": "A company's profit increased from 20 lakhs to 25 lakhs. What is the percentage increase?",
            "option_a": "20%",
            "option_b": "25%",
            "option_c": "30%",
            "option_d": "15%",
            "correct_answer": "B",
            "explanation": "Increase = 5, Percentage = (5/20) × 100 = 25%",
            "time_limit": 60,
            "year_asked": 2023
        },
        {
            "company_id": companies.get("Accenture"),
            "category": "Verbal Ability",
            "difficulty": "Easy",
            "question": "Choose the correct synonym for AMBIGUOUS:",
            "option_a": "Clear",
            "option_b": "Vague",
            "option_c": "Precise",
            "option_d": "Definite",
            "correct_answer": "B",
            "explanation": "Ambiguous = unclear or having multiple meanings → Vague",
            "time_limit": 45,
            "year_asked": 2024
        },
        {
            "company_id": companies.get("Cognizant"),
            "category": "Quantitative Aptitude",
            "difficulty": "Medium",
            "question": "A train travels 60 km in 45 minutes. What is its speed in km/hr?",
            "option_a": "70 km/hr",
            "option_b": "75 km/hr",
            "option_c": "80 km/hr",
            "option_d": "85 km/hr",
            "correct_answer": "C",
            "explanation": "Speed = 60 / (45/60) = 80 km/hr",
            "time_limit": 60,
            "year_asked": 2023
        }
    ]
    
    for q in aptitude_questions:
        c.execute("""
            INSERT INTO aptitude_questions (
                company_id, category, difficulty, question,
                option_a, option_b, option_c, option_d,
                correct_answer, explanation, time_limit, year_asked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            q["company_id"], q["category"], q["difficulty"], q["question"],
            q["option_a"], q["option_b"], q["option_c"], q["option_d"],
            q["correct_answer"], q["explanation"], q["time_limit"], q["year_asked"]
        ))
    
    conn.commit()
    print(f"✅ Added {len(aptitude_questions)} aptitude questions")
    conn.close()


def populate_technical_questions():
    """Add coding questions with test cases"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, name FROM companies")
    companies = {name: id for id, name in c.fetchall()}
    
    technical_questions = [
        {
            "company_id": companies.get("Amazon"),
            "difficulty": "Medium",
            "question_title": "Two Sum Problem",
            "question_description": "Find indices of two numbers that add up to a target.",
            "input_format": "n (array size), array elements, target",
            "output_format": "Indices (0-based)",
            "constraints": "2 ≤ n ≤ 10⁴",
            "sample_input": "4\n2 7 11 15\n9",
            "sample_output": "0 1",
            "test_cases": json.dumps([
                {"input": "4\n2 7 11 15\n9", "output": "0 1"},
                {"input": "3\n3 2 4\n6", "output": "1 2"},
                {"input": "2\n3 3\n6", "output": "0 1"}
            ]),
            "solution_code": "# Python Solution\ndef twoSum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        if target - num in seen:\n            return [seen[target - num], i]\n        seen[num] = i",
            "time_limit": 1800,
            "year_asked": 2024,
            "tags": "Array,Hash Table"
        },
        {
            "company_id": companies.get("Infosys"),
            "difficulty": "Easy",
            "question_title": "Reverse a String",
            "question_description": "Reverse a given string.",
            "input_format": "A single string",
            "output_format": "Reversed string",
            "constraints": "1 ≤ len(s) ≤ 1000",
            "sample_input": "Hello",
            "sample_output": "olleH",
            "test_cases": json.dumps([
                {"input": "Hello", "output": "olleH"},
                {"input": "Python", "output": "nohtyP"},
                {"input": "12345", "output": "54321"}
            ]),
            "solution_code": "# Python Solution\ndef reverse_string(s):\n    return s[::-1]",
            "time_limit": 900,
            "year_asked": 2023,
            "tags": "String"
        }
    ]
    
    for q in technical_questions:
        c.execute("""
            INSERT INTO technical_questions (
                company_id, difficulty, question_title, question_description,
                input_format, output_format, constraints, sample_input,
                sample_output, test_cases, solution_code, time_limit,
                year_asked, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            q["company_id"], q["difficulty"], q["question_title"],
            q["question_description"], q["input_format"], q["output_format"],
            q["constraints"], q["sample_input"], q["sample_output"],
            q["test_cases"], q["solution_code"], q["time_limit"],
            q["year_asked"], q["tags"]
        ))
    
    conn.commit()
    print(f"✅ Added {len(technical_questions)} technical questions")
    conn.close()


def populate_gd_topics():
    """Add Group Discussion topics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, name FROM companies")
    companies = {name: id for id, name in c.fetchall()}
    
    gd_topics = [
        {
            "company_id": companies.get("Capgemini"),
            "topic": "Artificial Intelligence: Boon or Bane for Employment",
            "category": "Technology",
            "description": "Impact of AI on jobs and human work.",
            "key_points": json.dumps([
                "Automation & job loss",
                "AI skill demand",
                "New job creation",
                "Ethics and responsibility"
            ]),
            "evaluation_criteria": "Content, Communication, Leadership, Listening, Teamwork",
            "time_limit": 600,
            "year_asked": 2024
        }
    ]
    
    for t in gd_topics:
        c.execute("""
            INSERT INTO gd_topics (
                company_id, topic, category, description,
                key_points, evaluation_criteria, time_limit, year_asked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t["company_id"], t["topic"], t["category"], t["description"],
            t["key_points"], t["evaluation_criteria"], t["time_limit"], t["year_asked"]
        ))
    
    conn.commit()
    print(f"✅ Added {len(gd_topics)} GD topics")
    conn.close()


def populate_hr_questions():
    """Add HR interview questions"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT id, name FROM companies")
    companies = {name: id for id, name in c.fetchall()}
    
    hr_questions = [
        {
            "company_id": companies.get("TCS"),
            "question": "Tell me about yourself.",
            "category": "Introduction",
            "sample_answer": "Brief background → Education → Key projects → Strengths → Career goal aligning with company vision.",
            "evaluation_points": json.dumps(["Confidence", "Relevance", "Clarity", "Structure"]),
            "difficulty": "Easy",
            "year_asked": 2024
        },
        {
            "company_id": companies.get("Infosys"),
            "question": "Why do you want to join Infosys?",
            "category": "Company Knowledge",
            "sample_answer": "Appreciate Infosys’ innovation culture, mention your skills fit, and career growth opportunities.",
            "evaluation_points": json.dumps(["Company knowledge", "Authenticity", "Career alignment"]),
            "difficulty": "Medium",
            "year_asked": 2023
        },
        {
            "company_id": companies.get("Amazon"),
            "question": "Describe a conflict in a team and how you handled it.",
            "category": "Behavioral",
            "sample_answer": "Use STAR method: Situation → Task → Action → Result. Show leadership and collaboration.",
            "evaluation_points": json.dumps(["Problem-solving", "Teamwork", "Communication", "Leadership"]),
            "difficulty": "Hard",
            "year_asked": 2023
        }
    ]
    
    for q in hr_questions:
        c.execute("""
            INSERT INTO hr_questions (
                company_id, question, category, sample_answer,
                evaluation_points, difficulty, year_asked
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            q["company_id"], q["question"], q["category"],
            q["sample_answer"], q["evaluation_points"],
            q["difficulty"], q["year_asked"]
        ))
    
    conn.commit()
    print(f"✅ Added {len(hr_questions)} HR questions")
    conn.close()


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM companies")
    if c.fetchone()[0] == 0:
        print("❌ No companies found. Run company_data_setup.py first!")
        conn.close()
        exit()
    conn.close()

    print("🚀 Starting question population...")
    populate_aptitude_questions()
    populate_technical_questions()
    populate_gd_topics()
    populate_hr_questions()
    print("\n🎯 All question banks populated successfully!")
    print("✅ Database ready for interview simulation.")
