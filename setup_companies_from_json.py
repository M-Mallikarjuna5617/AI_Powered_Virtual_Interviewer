"""
Setup Company Database from JSON Files
This script imports all companies and questions from your JSON files in company_datasets folder
"""

import sqlite3
import json
import os

DB_PATH = "questions.db"
COMPANY_DATASETS_FOLDER = "company_datasets"


def clear_existing_data():
    """Clear all existing company data safely (ignore missing tables)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print("🗑️  Clearing existing company data...")

    tables_to_clear = [
        "selected_companies",
        "feedback_reports",
        "hr_results",
        "gd_results",
        "technical_results",
        "aptitude_responses",
        "aptitude_attempts",
        "hr_questions",
        "gd_topics",
        "technical_questions",
        "aptitude_questions",
        "companies",
    ]

    for table in tables_to_clear:
        try:
            c.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            # Table doesn’t exist yet — ignore and continue
            pass

    conn.commit()
    conn.close()
    print("✅ Existing data cleared!")


def setup_tables():
    """Ensure all required tables exist (matching question_generator schema)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Companies
    c.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)

    # Aptitude (MCQ structure; fill unknowns with NULL/defaults)
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

    # Technical (coding)
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

    # GD Topics
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

    # HR Questions
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            question TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database tables (questions.db) ready!")


def import_company_from_json(json_file_path):
    """Import company data and questions from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    company_name = data['company']

    # Company details mapping
    company_details = {
        "Accenture": {
            "min_cgpa": 6.5,
            "required_skills": "Java,Python,Cloud Computing,Data Structures,Communication Skills",
            "role": "Associate Software Engineer",
            "package": "4.5-5.5 LPA",
            "location": "Bangalore/Pune/Hyderabad",
            "industry": "Consulting",
        },
        "Amazon": {
            "min_cgpa": 7.0,
            "required_skills": "Data Structures,Algorithms,Java,Python,System Design,Problem Solving",
            "role": "Software Development Engineer",
            "package": "18-28 LPA",
            "location": "Bangalore/Hyderabad",
            "industry": "E-commerce/Tech",
        },
        "Capgemini": {
            "min_cgpa": 6.5,
            "required_skills": "Java,Python,Testing,SQL,Problem Solving",
            "role": "Analyst",
            "package": "4.0-5.0 LPA",
            "location": "Bangalore/Mumbai/Pune",
            "industry": "Consulting",
        },
        "Cognizant": {
            "min_cgpa": 6.5,
            "required_skills": "Java,Python,SQL,Problem Solving,Analytical Skills",
            "role": "Programmer Analyst",
            "package": "4.0-4.5 LPA",
            "location": "Bangalore/Chennai/Hyderabad",
            "industry": "IT Services",
        },
        "HCL": {
            "min_cgpa": 6.0,
            "required_skills": "Java,C++,Python,SQL,Testing",
            "role": "Software Engineer",
            "package": "3.5-4.5 LPA",
            "location": "Bangalore/Noida/Chennai",
            "industry": "IT Services",
        },
        "IBM": {
            "min_cgpa": 7.0,
            "required_skills": "Java,Python,Cloud Computing,AI,Data Structures,Problem Solving",
            "role": "Application Developer",
            "package": "4.5-6.0 LPA",
            "location": "Bangalore/Pune/Hyderabad",
            "industry": "Technology",
        },
        "Infosys": {
            "min_cgpa": 6.0,
            "required_skills": "Java,Python,SQL,Data Structures,Problem Solving",
            "role": "System Engineer",
            "package": "3.5-4.5 LPA",
            "location": "Bangalore/Mysore/Pune",
            "industry": "IT Services",
        },
        "TCS": {
            "min_cgpa": 6.0,
            "required_skills": "C,C++,Java,SQL,Logical Reasoning",
            "role": "Assistant System Engineer",
            "package": "3.36 LPA",
            "location": "Multiple Locations",
            "industry": "IT Services",
        },
        "Tech Mahindra": {
            "min_cgpa": 6.0,
            "required_skills": "C,C++,Java,Testing,SQL,Communication",
            "role": "Software Engineer",
            "package": "3.25-3.75 LPA",
            "location": "Bangalore/Pune/Hyderabad",
            "industry": "IT Services",
        },
        "Wipro": {
            "min_cgpa": 6.0,
            "required_skills": "Java,C++,Python,OOPS,Database Management",
            "role": "Project Engineer",
            "package": "3.5-4.0 LPA",
            "location": "Bangalore/Hyderabad/Pune",
            "industry": "IT Services",
        },
    }

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Ensure company exists in questions.db companies
    c.execute("INSERT OR IGNORE INTO companies (name) VALUES (?)", (company_name,))
    c.execute("SELECT id FROM companies WHERE name = ?", (company_name,))
    row = c.fetchone()
    company_id = row[0]

    # Import aptitude
    aptitude_count = 0
    if 'aptitude' in data['rounds']:
        for q in data['rounds']['aptitude']:
            qtext = q.get('question', '')
            diff = q.get('difficulty', 'Medium')

            # Heuristic MCQ option generation
            import re
            numbers = re.findall(r"-?\d+", qtext)
            option_a = None
            option_b = None
            option_c = None
            option_d = None
            correct_answer = None  # Unknown from dataset

            if numbers:
                try:
                    base = int(numbers[-1])
                    option_a = str(base)
                    option_b = str(base + 1)
                    option_c = str(max(base - 1, 0))
                    option_d = str(base + 2)
                except Exception:
                    option_a, option_b, option_c, option_d = "A", "B", "C", "D"
            else:
                option_a, option_b, option_c, option_d = (
                    "Yes",
                    "No",
                    "Cannot be determined",
                    "None of the above",
                )

            c.execute(
                """
                INSERT INTO aptitude_questions (
                    company_id, category, difficulty, question,
                    option_a, option_b, option_c, option_d,
                    correct_answer, explanation, time_limit, year_asked
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id,
                    None,
                    diff,
                    qtext,
                    option_a,
                    option_b,
                    option_c,
                    option_d,
                    correct_answer,
                    None,
                    60,
                    2024,
                ),
            )
            aptitude_count += 1

    # Import technical
    technical_count = 0
    if 'technical' in data['rounds']:
        for q in data['rounds']['technical']:
            c.execute(
                """
                INSERT INTO technical_questions (
                    company_id, difficulty, question_title, question_description,
                    input_format, output_format, constraints, sample_input, sample_output,
                    test_cases, time_limit, year_asked, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id,
                    q.get('difficulty', 'Medium'),
                    q.get('question', ''),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    json.dumps([]),
                    180,
                    2024,
                    None,
                ),
            )
            technical_count += 1

    # Import GD
    gd_count = 0
    if 'gd' in data['rounds']:
        for topic in data['rounds']['gd']:
            c.execute(
                """
                INSERT INTO gd_topics (
                    company_id, topic, category, description, key_points, evaluation_criteria, time_limit, year_asked
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id,
                    topic.get('topic', ''),
                    'General',
                    topic.get('topic', ''),
                    None,
                    None,
                    180,
                    2024,
                ),
            )
            gd_count += 1

    # Import HR
    hr_count = 0
    if 'hr' in data['rounds']:
        for q in data['rounds']['hr']:
            c.execute(
                """
                INSERT INTO hr_questions (company_id, question)
                VALUES (?, ?)
                """,
                (company_id, q.get('question', '')),
            )
            hr_count += 1

    conn.commit()
    conn.close()

    print(f"✅ Imported {company_name}:")
    print(f"   - Aptitude: {aptitude_count} questions")
    print(f"   - Technical: {technical_count} questions")
    print(f"   - GD Topics: {gd_count} topics")
    print(f"   - HR Questions: {hr_count} questions")


def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 Setting up Company Database from JSON Files")
    print("=" * 60)

    if not os.path.exists(COMPANY_DATASETS_FOLDER):
        print(f"❌ Error: '{COMPANY_DATASETS_FOLDER}' folder not found!")
        print("💡 Please ensure the folder exists with all JSON files")
        return

    # Step 1: Setup tables first (to avoid missing table errors)
    setup_tables()

    # Step 2: Then clear data safely
    clear_existing_data()

    # Step 3: Import all JSONs
    json_files = [f for f in os.listdir(COMPANY_DATASETS_FOLDER) if f.endswith('.json')]

    if not json_files:
        print(f"❌ No JSON files found in '{COMPANY_DATASETS_FOLDER}' folder!")
        return

    print(f"\n📥 Found {len(json_files)} JSON files. Importing...")
    print("-" * 60)

    for json_file in sorted(json_files):
        file_path = os.path.join(COMPANY_DATASETS_FOLDER, json_file)
        try:
            import_company_from_json(file_path)
        except Exception as e:
            print(f"❌ Error importing {json_file}: {e}")

    # Step 4: Summary
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM companies")
    total_companies = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM aptitude_questions")
    total_aptitude = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM technical_questions")
    total_technical = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM gd_topics")
    total_gd = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM hr_questions")
    total_hr = c.fetchone()[0]

    c.execute("SELECT name FROM companies ORDER BY name")
    companies = [row[0] for row in c.fetchall()]

    conn.close()

    print("\n" + "=" * 60)
    print("📊 IMPORT SUMMARY")
    print("=" * 60)
    print(f"✅ Total Companies: {total_companies}")
    print(f"   Companies: {', '.join(companies)}")
    print(f"\n✅ Total Aptitude Questions: {total_aptitude}")
    print(f"✅ Total Technical Questions: {total_technical}")
    print(f"✅ Total GD Topics: {total_gd}")
    print(f"✅ Total HR Questions: {total_hr}")
    print("=" * 60)
    print("\n🎉 Database setup complete!")
    print("💡 Next: Run your Flask app to start using the system")
    print("   Command: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
