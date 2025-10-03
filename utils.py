import os
import sqlite3

# -------------------- Database Connection --------------------
def get_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# -------------------- Send Email (OTP demo) --------------------
def send_email(receiver_email, otp):
    # Dev mode: just print OTP to console and log it
    print(f"[OTP] To: {receiver_email} | OTP: {otp}")
    with open("otps.log", "a") as f:
        f.write(f"{receiver_email}\t{otp}\n")
    return True

# -------------------- Parse Resume --------------------
def parse_resume(file_path, email):
    """
    Minimal demo: parse resume and update student record
    For production, you can integrate PyPDF2 / docx / NLP libraries
    """
    student_name = os.path.splitext(os.path.basename(file_path))[0]  # use file name as name
    conn = get_connection()
    c = conn.cursor()

    # If student exists, update; else insert
    c.execute("SELECT * FROM students WHERE email=?", (email,))
    if c.fetchone():
        c.execute("UPDATE students SET name=? WHERE email=?", (student_name, email))
    else:
        c.execute("INSERT INTO students (name, email, skills, cgpa, graduation_year) VALUES (?, ?, ?, ?, ?)",
                  (student_name, email, "", 0.0, 2025))

    conn.commit()
    conn.close()

    return {"name": student_name, "email": email}

# -------------------- Match Companies --------------------
def match_companies(email):
    conn = get_connection()
    c = conn.cursor()

    # Fetch student
    c.execute("SELECT * FROM students WHERE email=?", (email,))
    student = c.fetchone()
    if not student:
        return []

    student_skills = set([s.strip().lower() for s in (student["skills"] or "").split(",")])
    cgpa = student["cgpa"]
    grad_year = student["graduation_year"]

    # Fetch companies
    c.execute("SELECT * FROM companies")
    companies = c.fetchall()
    conn.close()

    eligible = []
    for comp in companies:
        required_skills = set([s.strip().lower() for s in (comp["required_skills"] or "").split(",")])
        if (cgpa >= comp["min_cgpa"] and
            grad_year == comp["graduation_year"] and
            required_skills.issubset(student_skills)):
            eligible.append(dict(comp))

    return eligible
