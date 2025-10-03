import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT,
            provider TEXT DEFAULT 'local'
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def add_user(name, email, password=None, provider="local"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    hashed_password = generate_password_hash(password) if password else None
    cursor.execute(
        "INSERT INTO users (name, email, password, provider) VALUES (?, ?, ?, ?)",
        (name, email, hashed_password, provider)
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_user(email):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, email, password FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user:
        return {"name": user[0], "email": user[1], "password": user[2]}
    return None
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_resume_tables():
    conn = get_connection()
    c = conn.cursor()
    
    # Students table
    c.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        name TEXT,
        cgpa REAL,
        graduation_year INTEGER,
        skills TEXT
    )
    """)
    
    # Resumes table
    c.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        file_path TEXT
    )
    """)
    
    # Companies table
    c.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        min_cgpa REAL,
        required_skills TEXT,
        graduation_year INTEGER,
        role TEXT,
        package_offered TEXT,
        location TEXT
    )
    """)
    
    conn.commit()
    conn.close()