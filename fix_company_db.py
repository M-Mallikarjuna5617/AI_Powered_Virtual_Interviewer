"""
Enhanced Company Database Setup for Karnataka-based Companies
This script initializes comprehensive company data with real placement patterns
"""

import sqlite3
from datetime import datetime

DB_PATH = "users.db"

def setup_company_database():
    """Setup comprehensive company database with Karnataka focus"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Drop and recreate companies table with enhanced fields
    c.execute("DROP TABLE IF EXISTS companies")
    c.execute("""
        CREATE TABLE companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            min_cgpa REAL DEFAULT 0.0,
            required_skills TEXT,
            graduation_year INTEGER,
            role TEXT,
            package_offered TEXT,
            location TEXT,
            industry TEXT,
            eligibility_criteria TEXT,
            selection_process TEXT,
            no_of_rounds INTEGER DEFAULT 4,
            active_backlogs INTEGER DEFAULT 0,
            company_description TEXT,
            last_visited_year INTEGER
        )
    """)
    
    # Karnataka-based companies with comprehensive data
    companies = [
        # Tech Giants with Bangalore offices
        {
            "name": "Infosys",
            "min_cgpa": 6.0,
            "required_skills": "Java,Python,SQL,Data Structures,Problem Solving",
            "graduation_year": 2025,
            "role": "System Engineer",
            "package_offered": "3.5-4.5 LPA",
            "location": "Bangalore/Mysore",
            "industry": "IT Services",
            "eligibility_criteria": "B.E/B.Tech (CS/IS/EC/EE) | No active backlogs | 60% in 10th, 12th, and Degree",
            "selection_process": "Online Test → Technical Interview → HR Interview",
            "no_of_rounds": 3,
            "active_backlogs": 0,
            "company_description": "Global leader in consulting, technology and outsourcing solutions",
            "last_visited_year": 2024
        },
        {
            "name": "Wipro",
            "min_cgpa": 6.0,
            "required_skills": "Java,C++,Python,OOPS,Database Management",
            "graduation_year": 2025,
            "role": "Project Engineer",
            "package_offered": "3.5-4.0 LPA",
            "location": "Bangalore",
            "industry": "IT Services",
            "eligibility_criteria": "All Engineering branches | Max 2 active backlogs | 60% throughout",
            "selection_process": "Aptitude Test → Technical Test → Interview → HR Round",
            "no_of_rounds": 4,
            "active_backlogs": 2,
            "company_description": "Leading global information technology, consulting and business process services",
            "last_visited_year": 2024
        },
        {
            "name": "TCS",
            "min_cgpa": 6.0,
            "required_skills": "C,C++,Java,SQL,Logical Reasoning",
            "graduation_year": 2025,
            "role": "Assistant System Engineer",
            "package_offered": "3.36 LPA",
            "location": "Bangalore/Multiple",
            "industry": "IT Services",
            "eligibility_criteria": "BE/BTech/MCA/MSc | No active backlogs during selection",
            "selection_process": "National Qualifier Test → Technical Interview → Managerial Round → HR",
            "no_of_rounds": 4,
            "active_backlogs": 0,
            "company_description": "India's largest IT services, consulting and business solutions organization",
            "last_visited_year": 2024
        },
        {
            "name": "Accenture",
            "min_cgpa": 6.5,
            "required_skills": "Java,Python,Cloud Computing,Data Structures,Communication Skills",
            "graduation_year": 2025,
            "role": "Associate Software Engineer",
            "package_offered": "4.5-5.5 LPA",
            "location": "Bangalore",
            "industry": "Consulting",
            "eligibility_criteria": "BE/BTech (all branches) | 65% in X, XII and Graduation | No backlogs",
            "selection_process": "Cognitive Assessment → Coding Test → Technical Interview → HR Interview",
            "no_of_rounds": 4,
            "active_backlogs": 0,
            "company_description": "Global professional services company with leading capabilities in digital, cloud and security",
            "last_visited_year": 2024
        },
        {
            "name": "Cognizant",
            "min_cgpa": 6.5,
            "required_skills": "Java,Python,SQL,Problem Solving,Analytical Skills",
            "graduation_year": 2025,
            "role": "Programmer Analyst",
            "package_offered": "4.0-4.5 LPA",
            "location": "Bangalore",
            "industry": "IT Services",
            "eligibility_criteria": "BE/BTech (CS/IS/IT/EC/EE) | 60% throughout | Max 1 backlog",
            "selection_process": "Online Test → Technical Interview → HR Round",
            "no_of_rounds": 3,
            "active_backlogs": 1,
            "company_description": "One of the world's leading professional services companies",
            "last_visited_year": 2024
        },
        {
            "name": "Tech Mahindra",
            "min_cgpa": 6.0,
            "required_skills": "C,C++,Java,Testing,SQL,Communication",
            "graduation_year": 2025,
            "role": "Software Engineer",
            "package_offered": "3.25-3.75 LPA",
            "location": "Bangalore",
            "industry": "IT Services",
            "eligibility_criteria": "BE/BTech (all branches) | 60% in X, XII, Degree | No active backlogs",
            "selection_process": "Online Test → Technical Interview → HR Interview",
            "no_of_rounds": 3,
            "active_backlogs": 0,
            "company_description": "Specialized in digital transformation, consulting and business re-engineering",
            "last_visited_year": 2024
        },
        {
            "name": "Mindtree",
            "min_cgpa": 6.5,
            "required_skills": "Java,Python,Full Stack Development,Data Structures,Algorithms",
            "graduation_year": 2025,
            "role": "Software Engineer",
            "package_offered": "3.5-4.2 LPA",
            "location": "Bangalore",
            "industry": "IT Services",
            "eligibility_criteria": "BE/BTech (CS/IS/IT) | 65% throughout | No backlogs",
            "selection_process": "Aptitude → Coding Test → Technical Interview → HR",
            "no_of_rounds": 4,
            "active_backlogs": 0,
            "company_description": "Digital transformation and technology services company",
            "last_visited_year": 2024
        },
        {
            "name": "Mphasis",
            "min_cgpa": 6.0,
            "required_skills": "Java,Python,Cloud,Data Structures,SQL",
            "graduation_year": 2025,
            "role": "Associate Software Engineer",
            "package_offered": "3.5-4.0 LPA",
            "location": "Bangalore",
            "industry": "IT Services",
            "eligibility_criteria": "BE/BTech (all branches) | 60% throughout | Max 2 backlogs",
            "selection_process": "Online Assessment → Technical Round → HR Round",
            "no_of_rounds": 3,
            "active_backlogs": 2,
            "company_description": "IT solutions provider specializing in cloud and cognitive services",
            "last_visited_year": 2023
        },
        {
            "name": "Capgemini",
            "min_cgpa": 6.5,
            "required_skills": "Java,Python,Testing,SQL,Problem Solving",
            "graduation_year": 2025,
            "role": "Analyst",
            "package_offered": "4.0-5.0 LPA",
            "location": "Bangalore",
            "industry": "Consulting",
            "eligibility_criteria": "BE/BTech (all branches) | 60% in X, XII, Degree | No active backlogs",
            "selection_process": "Online Test → Group Discussion → Technical Interview → HR",
            "no_of_rounds": 4,
            "active_backlogs": 0,
            "company_description": "Global leader in consulting, technology services and digital transformation",
            "last_visited_year": 2024
        },
        {
            "name": "IBM India",
            "min_cgpa": 7.0,
            "required_skills": "Java,Python,Cloud Computing,AI,Data Structures,Problem Solving",
            "graduation_year": 2025,
            "role": "Application Developer",
            "package_offered": "4.5-6.0 LPA",
            "location": "Bangalore",
            "industry": "Technology",
            "eligibility_criteria": "BE/BTech (CS/IS/IT/EC) | 65% throughout | No backlogs",
            "selection_process": "Cognitive Assessment → Coding Test → Technical Interview → Managerial Round",
            "no_of_rounds": 4,
            "active_backlogs": 0,
            "company_description": "Leading hybrid cloud and AI, and consulting expertise",
            "last_visited_year": 2024
        },
        {
            "name": "Oracle",
            "min_cgpa": 7.5,
            "required_skills": "Java,SQL,Database Management,Cloud,Data Structures,Algorithms",
            "graduation_year": 2025,
            "role": "Associate Consultant",
            "package_offered": "6.0-8.0 LPA",
            "location": "Bangalore",
            "industry": "Technology",
            "eligibility_criteria": "BE/BTech (CS/IS/IT) | 70% throughout | No backlogs",
            "selection_process": "Online Test → Technical Interviews (2 rounds) → HR",
            "no_of_rounds": 4,
            "active_backlogs": 0,
            "company_description": "Integrated cloud applications and platform services",
            "last_visited_year": 2023
        },
        {
            "name": "Amazon",
            "min_cgpa": 7.0,
            "required_skills": "Data Structures,Algorithms,Java,Python,System Design,Problem Solving",
            "graduation_year": 2025,
            "role": "Software Development Engineer",
            "package_offered": "18-28 LPA",
            "location": "Bangalore",
            "industry": "E-commerce/Tech",
            "eligibility_criteria": "BE/BTech (CS/IS/IT/EC) | Strong coding skills | No backlogs",
            "selection_process": "Online Test → Technical Interviews (3-4 rounds) → HR/Bar Raiser",
            "no_of_rounds": 5,
            "active_backlogs": 0,
            "company_description": "World's largest online retailer and cloud computing platform",
            "last_visited_year": 2024
        }
    ]
    
    # Insert companies
    for comp in companies:
        c.execute("""
            INSERT INTO companies (
                name, min_cgpa, required_skills, graduation_year, role, 
                package_offered, location, industry, eligibility_criteria,
                selection_process, no_of_rounds, active_backlogs,
                company_description, last_visited_year
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            comp["name"], comp["min_cgpa"], comp["required_skills"], 
            comp["graduation_year"], comp["role"], comp["package_offered"],
            comp["location"], comp["industry"], comp["eligibility_criteria"],
            comp["selection_process"], comp["no_of_rounds"], comp["active_backlogs"],
            comp["company_description"], comp["last_visited_year"]
        ))
    
    conn.commit()
    conn.close()
    print("✅ Enhanced company database created successfully!")


def create_question_banks():
    """Create comprehensive question banks for all companies"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Aptitude Questions Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS aptitude_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            category TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            question TEXT NOT NULL,
            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,
            correct_answer TEXT NOT NULL,
            explanation TEXT,
            time_limit INTEGER DEFAULT 60,
            year_asked INTEGER,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    
    # Technical Questions Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS technical_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            difficulty TEXT NOT NULL,
            question_title TEXT NOT NULL,
            question_description TEXT NOT NULL,
            input_format TEXT,
            output_format TEXT,
            constraints TEXT,
            sample_input TEXT,
            sample_output TEXT,
            test_cases TEXT,
            solution_code TEXT,
            time_limit INTEGER DEFAULT 1800,
            year_asked INTEGER,
            tags TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    
    # GD Topics Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS gd_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            topic TEXT NOT NULL,
            category TEXT,
            description TEXT,
            key_points TEXT,
            evaluation_criteria TEXT,
            time_limit INTEGER DEFAULT 600,
            year_asked INTEGER,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    
    # HR Questions Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            question TEXT NOT NULL,
            category TEXT,
            sample_answer TEXT,
            evaluation_points TEXT,
            difficulty TEXT,
            year_asked INTEGER,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    
    # Student Test History
    c.execute("""
        CREATE TABLE IF NOT EXISTS test_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_id INTEGER,
            test_type TEXT NOT NULL,
            score REAL,
            total_questions INTEGER,
            correct_answers INTEGER,
            time_taken INTEGER,
            feedback TEXT,
            test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Question bank tables created successfully!")


if __name__ == "__main__":
    setup_company_database()
    create_question_banks()
    print("\n🎉 Complete database setup finished!")
    print("Next: Run populate_questions.py to add question data")