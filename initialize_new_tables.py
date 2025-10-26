"""
Script to initialize new database tables for improved test tracking
Run this once to add the new tables to your existing database
"""

import sqlite3

DB_PATH = "users.db"

def create_new_tables():
    """Create new tables for improved test tracking"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("🔧 Initializing new database tables...")
    
    # Aptitude test attempts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aptitude_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            total_questions INTEGER DEFAULT 30,
            correct_answers INTEGER DEFAULT 0,
            score REAL DEFAULT 0,
            time_taken INTEGER,
            status TEXT DEFAULT 'completed',
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✅ aptitude_attempts table created")
    
    # Aptitude test responses
    cursor.execute("""
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
    print("✅ aptitude_responses table created")
    
    # Selected companies
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS selected_companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_id INTEGER NOT NULL,
            selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    print("✅ selected_companies table created")
    
    # Question bank (if it doesn't exist)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_bank (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text TEXT NOT NULL,
            option_a TEXT,
            option_b TEXT,
            option_c TEXT,
            option_d TEXT,
            correct_answer TEXT,
            explanation TEXT,
            difficulty TEXT,
            subtopic TEXT,
            time_limit INTEGER DEFAULT 60
        )
    """)
    print("✅ question_bank table checked")
    
    # Insert sample questions if question_bank is empty
    cursor.execute("SELECT COUNT(*) FROM question_bank")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("📝 Adding sample aptitude questions...")
        sample_questions = [
            ("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?", 
             "Yes", "No", "Cannot be determined", "Insufficient data", "A",
             "Valid syllogism: A⊆B and B⊆C implies A⊆C", "Easy", "Logical Reasoning"),
            ("What is 15% of 200?", "25", "30", "35", "40", "B",
             "15/100 × 200 = 30", "Easy", "Quantitative Aptitude"),
            ("Synonym of AMBIGUOUS:", "Clear", "Vague", "Precise", "Definite", "B",
             "Ambiguous means unclear", "Easy", "Verbal Ability"),
            ("Find the next number: 2, 6, 12, 20, 30, ?", "40", "42", "44", "46", "B",
             "Differences are 4, 6, 8, 10, 12. Next is 30 + 12 = 42", "Easy", "Number Series"),
            ("A train travels 60 km in 45 minutes. What is its speed in km/hr?", "70", "75", "80", "85", "C",
             "Speed = Distance/Time = 60/(45/60) = 80 km/hr", "Medium", "Quantitative Aptitude"),
            ("Profit increased from 20L to 25L. Percentage increase?", "20%", "25%", "30%", "15%", "B",
             "Increase = 5 lakhs. Percentage = (5/20) × 100 = 25%", "Medium", "Data Interpretation"),
            ("If South-East becomes North and North becomes South-West, what does West become?", 
             "North-East", "North-West", "South-East", "South", "A",
             "135° clockwise rotation. West becomes North-East", "Hard", "Logical Reasoning"),
            ("Complete: 'He is allergic ___ dust'", "at", "to", "from", "with", "B",
             "Allergic to is correct usage", "Easy", "Verbal Ability"),
            ("Simple Interest on Rs.1000 at 10% for 2 years?", "100", "150", "200", "250", "C",
             "SI = PRT/100 = 1000×10×2/100 = 200", "Easy", "Quantitative Aptitude"),
            ("Find missing number: 1, 4, 9, 16, ?, 36", "20", "25", "30", "32", "B",
             "These are perfect squares: 1², 2², 3², 4², 5², 6². Missing is 5² = 25", "Hard", "Number Series")
        ]
        
        cursor.executemany("""
            INSERT INTO question_bank 
            (question_text, option_a, option_b, option_c, option_d, correct_answer, explanation, difficulty, subtopic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_questions)
        print(f"✅ Added {len(sample_questions)} sample questions")
    
    conn.commit()
    conn.close()
    
    print("\n🎉 Database initialization complete!")
    print("✅ All new tables created successfully")
    print("✅ Sample questions added to question_bank")

if __name__ == "__main__":
    try:
        create_new_tables()
        print("\n✨ You can now run the application with improved test tracking!")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        print("Please check your database file and permissions.")

