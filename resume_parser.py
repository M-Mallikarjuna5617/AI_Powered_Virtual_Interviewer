import pdfplumber
import re
import mysql.connector
from database import get_connection

# === DB Connection ===
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yourpassword",
    database="virtual_interviewer"
)
cursor = db.cursor()

def parse_resume(pdf_path, student_id=1):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    # --- Regex Patterns ---
    name = re.search(r"Name[:\- ]*(.*)", text, re.I)
    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    cgpa = re.search(r"(\d\.\d{1,2})\s*CGPA", text, re.I)
    year = re.search(r"(20\d{2})", text)   # finds graduation year like 2026
    skills_section = re.findall(r"(Python|Java|C\+\+|SQL|Machine Learning|AI|Cloud)", text, re.I)

    # --- Extracted Info ---
    student_name = name.group(1).strip() if name else "Unknown"
    student_email = email.group(0) if email else "unknown@email.com"
    student_cgpa = float(cgpa.group(1)) if cgpa else 0.0
    grad_year = int(year.group(1)) if year else 2026
    student_skills = ",".join(set([s.capitalize() for s in skills_section])) if skills_section else "Not Found"

    print("Extracted Data:", student_name, student_email, student_cgpa, grad_year, student_skills)

    # --- Insert into DB ---
    sql = """INSERT INTO students (id, name, email, cgpa, graduation_year, skills)
             VALUES (%s, %s, %s, %s, %s, %s)
             ON DUPLICATE KEY UPDATE name=%s, email=%s, cgpa=%s, graduation_year=%s, skills=%s"""
    data = (student_id, student_name, student_email, student_cgpa, grad_year, student_skills,
            student_name, student_email, student_cgpa, grad_year, student_skills)
    
    cursor.execute(sql, data)
    db.commit()
    print("✅ Student profile inserted/updated in DB")

# Example usage:
# parse_resume("uploads/sample_resume.pdf", student_id=1)
   # --- Save into DB ---
    conn = get_connection()
    c = conn.cursor()
    c.execute("""INSERT INTO students (email, name, cgpa, graduation_year, skills)
                 VALUES (?, ?, ?, ?, ?)
                 ON CONFLICT(email) DO UPDATE SET 
                     name=excluded.name, 
                     cgpa=excluded.cgpa, 
                     graduation_year=excluded.graduation_year, 
                     skills=excluded.skills""",
              (email, student_name, student_cgpa, grad_year, student_skills))
    conn.commit()
    conn.close()

    return {
        "name": student_name,
        "cgpa": student_cgpa,
        "graduation_year": grad_year,
        "skills": student_skills
    }