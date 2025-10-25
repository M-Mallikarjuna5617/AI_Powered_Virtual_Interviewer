import sqlite3
import os

DB_PATH = "users.db"

# ----------------- Init DB -----------------
def init_db():
    """Create all necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT,
            provider TEXT
        )
    """)

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            cgpa REAL DEFAULT 0.0,
            graduation_year INTEGER DEFAULT 2025,
            skills TEXT DEFAULT ''
        )
    """)

    # Resumes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            file_path TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # Companies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            min_cgpa REAL DEFAULT 0.0,
            required_skills TEXT,
            graduation_year INTEGER,
            role TEXT,
            package_offered TEXT,
            location TEXT,
            industry TEXT
        )
    """)

    # Technical questions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            domain TEXT NOT NULL,
            question TEXT NOT NULL,
            test_cases TEXT,
            difficulty TEXT DEFAULT 'medium',
            year INTEGER,
            language TEXT DEFAULT 'python',
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)

    # GD topics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gd_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            topic TEXT NOT NULL,
            description TEXT,
            year INTEGER,
            difficulty TEXT DEFAULT 'medium',
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)

    # HR questions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hr_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            question TEXT NOT NULL,
            category TEXT,
            difficulty TEXT DEFAULT 'medium',
            year INTEGER,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)

    # Technical interview results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            question_id INTEGER,
            code_submitted TEXT,
            language TEXT,
            test_results TEXT,
            score REAL,
            execution_time REAL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(question_id) REFERENCES technical_questions(id)
        )
    """)

    # GD results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gd_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            topic_id INTEGER,
            transcript TEXT,
            fluency_score REAL,
            clarity_score REAL,
            confidence_score REAL,
            overall_score REAL,
            feedback TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(topic_id) REFERENCES gd_topics(id)
        )
    """)

    # HR interview results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hr_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            question_id INTEGER,
            answer TEXT,
            clarity_score REAL,
            relevance_score REAL,
            confidence_score REAL,
            overall_score REAL,
            feedback TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(question_id) REFERENCES hr_questions(id)
        )
    """)

    # Comprehensive feedback reports
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_name TEXT,
            aptitude_score REAL,
            technical_score REAL,
            gd_score REAL,
            hr_score REAL,
            overall_score REAL,
            strengths TEXT,
            improvements TEXT,
            recommendations TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ----------------- Database Connection -----------------
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ----------------- Users Functions -----------------
def add_user(name, email, password=None, provider=None):
    """Add a new user to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO users (name, email, password, provider) VALUES (?, ?, ?, ?)",
            (name, email, password, provider)
        )
        conn.commit()
    except Exception as e:
        print(f"Error adding user: {e}")
    finally:
        conn.close()


def get_user(email):
    """Get user details by email."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password, provider FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "password": row[3],
            "provider": row[4]
        }
    return None


# ----------------- Resume Functions -----------------
def save_resume(email, file_path):
    """Save or update a student's resume."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure student exists in students table
    cursor.execute("INSERT OR IGNORE INTO students (email) VALUES (?)", (email,))
    conn.commit()
    
    # Get student ID
    cursor.execute("SELECT id FROM students WHERE email = ?", (email,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return False
    
    student_id = result[0]

    # Check if resume already exists for this student
    cursor.execute("SELECT id FROM resumes WHERE student_id = ?", (student_id,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing resume
        cursor.execute(
            "UPDATE resumes SET file_path = ?, uploaded_at = CURRENT_TIMESTAMP WHERE student_id = ?",
            (file_path, student_id)
        )
    else:
        # Insert new resume
        cursor.execute(
            "INSERT INTO resumes (student_id, file_path) VALUES (?, ?)",
            (student_id, file_path)
        )

    conn.commit()
    conn.close()
    return True


def get_resume_by_email(email):
    """Return the full file path of a student's resume."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.file_path
        FROM resumes r
        JOIN students s ON r.student_id = s.id
        WHERE s.email = ?
    """, (email,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


# ----------------- Student Functions -----------------
def get_student(email):
    """Get student details by email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE email = ?", (email,))
    student = cursor.fetchone()
    conn.close()
    return dict(student) if student else None


def update_student_profile(email, name=None, cgpa=None, graduation_year=None, skills=None):
    """Update student profile information."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    updates = []
    params = []
    
    if name:
        updates.append("name = ?")
        params.append(name)
    if cgpa is not None:
        updates.append("cgpa = ?")
        params.append(cgpa)
    if graduation_year:
        updates.append("graduation_year = ?")
        params.append(graduation_year)
    if skills:
        updates.append("skills = ?")
        params.append(skills)
    
    if updates:
        params.append(email)
        query = f"UPDATE students SET {', '.join(updates)} WHERE email = ?"
        cursor.execute(query, params)
        conn.commit()
    
    conn.close()