import pdfplumber
import re
import sqlite3
import os

DB_PATH = "users.db"

# -------------------- Parse Resume --------------------
def parse_resume(file_path, student_email):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

    # Extract using regex
    name = re.search(r"Name[:\- ]*(.*)", text, re.I)
    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    cgpa = re.search(r"(\d\.\d{1,2})\s*CGPA", text, re.I)
    year = re.search(r"(20\d{2})", text)
    skills_section = re.findall(r"(Python|Java|C\+\+|SQL|Machine Learning|AI|Cloud)", text, re.I)

    student_name = name.group(1).strip() if name else "Unknown"
    student_email = email.group(0) if email else student_email
    student_cgpa = float(cgpa.group(1)) if cgpa else 0.0
    grad_year = int(year.group(1)) if year else 2026
    student_skills = ",".join(set([s.capitalize() for s in skills_section])) if skills_section else "Not Found"

    # Save to DB (students table)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            email TEXT PRIMARY KEY,
            name TEXT,
            cgpa REAL,
            graduation_year INT,
            skills TEXT
        )
    """)
    c.execute("""
        INSERT OR REPLACE INTO students (email, name, cgpa, graduation_year, skills)
        VALUES (?, ?, ?, ?, ?)
    """, (student_email, student_name, student_cgpa, grad_year, student_skills))
    conn.commit()
    conn.close()

    return {
        "name": student_name,
        "email": student_email,
        "cgpa": student_cgpa,
        "graduation_year": grad_year,
        "skills": student_skills
    }


# -------------------- Match Companies --------------------
def match_companies(student_email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get student details
    c.execute("SELECT * FROM students WHERE email=?", (student_email,))
    student = c.fetchone()
    if not student:
        return []

    _, name, cgpa, grad_year, skills = student
    student_skills = set([s.strip().lower() for s in skills.split(",")])

    # Create companies table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            min_cgpa REAL,
            required_skills TEXT,
            graduation_year INT,
            role TEXT,
            package_offered TEXT,
            location TEXT
        )
    """)

    # Sample companies if empty
    c.execute("SELECT COUNT(*) FROM companies")
    if c.fetchone()[0] == 0:
        c.executemany("""
            INSERT INTO companies (name, min_cgpa, required_skills, graduation_year, role, package_offered, location)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            ("Google", 8.0, "Python,Machine Learning,SQL", 2026, "Software Engineer", "25 LPA", "Bangalore"),
            ("TCS", 6.0, "Java,SQL", 2026, "System Engineer", "4 LPA", "Hyderabad"),
            ("Infosys", 6.5, "Python,Data Structures", 2026, "Associate Engineer", "5 LPA", "Pune"),
            ("Amazon", 7.5, "Java,Python,Cloud", 2026, "SDE 1", "20 LPA", "Bangalore")
        ])
        conn.commit()

    # Fetch companies
    c.execute("SELECT * FROM companies")
    companies = c.fetchall()

    eligible = []
    for comp in companies:
        _, cname, min_cgpa, req_skills, comp_year, role, package, location = comp
        required_skills = set([s.strip().lower() for s in req_skills.split(",")])

        if (cgpa >= min_cgpa and grad_year == comp_year and required_skills.issubset(student_skills)):
            eligible.append({
                "name": cname,
                "role": role,
                "package": package,
                "location": location
            })

    conn.close()
    return eligible
