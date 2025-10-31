import pdfplumber
import re
import sqlite3
import os
from database import get_connection

DB_PATH = "users.db"

# -------------------- Send Email (OTP demo) --------------------
def send_email(receiver_email, otp):
    """Send OTP email (development mode - prints to console)."""
    print(f"[OTP] To: {receiver_email} | OTP: {otp}")
    with open("otps.log", "a") as f:
        f.write(f"{receiver_email}\t{otp}\n")
    return True

# -------------------- Initialize Company Database --------------------
def init_company_data():
    """Initialize company database with real companies and their criteria."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if companies already exist
    c.execute("SELECT COUNT(*) FROM companies")
    if c.fetchone()[0] > 0:
        conn.close()
        return
    
    companies_data = [
        # Tech Giants
        ("Google", 8.0, "Python,Machine Learning,Data Structures,Algorithms", 2025, "Software Engineer", "25-30 LPA", "Bangalore", "Tech"),
        ("Microsoft", 7.5, "C++,Data Structures,Cloud Computing,Azure", 2025, "Software Developer", "20-25 LPA", "Hyderabad", "Tech"),
        ("Amazon", 7.5, "Java,Python,AWS,System Design", 2025, "SDE-1", "18-24 LPA", "Bangalore", "Tech"),
        ("Meta", 8.5, "Python,React,Data Structures,System Design", 2025, "Software Engineer", "28-35 LPA", "Bangalore", "Tech"),
        
        # Indian Tech Companies
        ("TCS", 6.0, "Java,SQL,C++", 2025, "Assistant System Engineer", "3.5-4 LPA", "Multiple", "IT Services"),
        ("Infosys", 6.5, "Python,Java,SQL", 2025, "System Engineer", "4-5 LPA", "Pune", "IT Services"),
        ("Wipro", 6.5, "Java,Python,Cloud", 2025, "Project Engineer", "3.5-4.5 LPA", "Bangalore", "IT Services"),
        ("Tech Mahindra", 6.0, "Java,Testing,SQL", 2025, "Software Engineer", "3.5-4 LPA", "Hyderabad", "IT Services"),
        
        # Product Companies
        ("Flipkart", 7.0, "Java,Python,System Design,Algorithms", 2025, "SDE-1", "12-18 LPA", "Bangalore", "E-commerce"),
        ("Swiggy", 7.0, "Python,Java,Data Structures", 2025, "Software Engineer", "10-15 LPA", "Bangalore", "Food Tech"),
        ("Paytm", 6.8, "Java,Python,SQL,API Development", 2025, "Software Developer", "8-12 LPA", "Noida", "Fintech"),
        ("Zomato", 7.0, "Python,Java,React", 2025, "Software Engineer", "10-14 LPA", "Gurgaon", "Food Tech"),
        
        # Startups
        ("Razorpay", 7.2, "Python,Java,Payment Systems,API", 2025, "Software Engineer", "12-18 LPA", "Bangalore", "Fintech"),
        ("CRED", 7.5, "Python,Kotlin,System Design", 2025, "Software Engineer", "15-20 LPA", "Bangalore", "Fintech"),
        ("Zerodha", 7.0, "Python,Trading Systems,Algorithms", 2025, "Software Developer", "12-16 LPA", "Bangalore", "Fintech"),
        
        # Consulting
        ("Deloitte", 7.0, "Data Analysis,SQL,Python,Business Intelligence", 2025, "Analyst", "6-8 LPA", "Multiple", "Consulting"),
        ("Accenture", 6.5, "Java,Cloud,Testing", 2025, "Associate Software Engineer", "4.5-6 LPA", "Multiple", "Consulting"),
        ("Cognizant", 6.0, "Java,Python,Testing", 2025, "Programmer Analyst", "4-5 LPA", "Multiple", "IT Services"),
    ]
    
    c.executemany("""
        INSERT INTO companies (name, min_cgpa, required_skills, graduation_year, role, package_offered, location, industry)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, companies_data)
    
    conn.commit()
    conn.close()
    print("✅ Company database initialized with real companies!")


# -------------------- Parse Resume --------------------
def parse_resume(file_path, student_email):
    """Extract information from resume PDF."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    # Extract information using regex
    name_match = re.search(r"(?:Name|NAME)[:\-\s]*([\w\s]+)", text, re.I)
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    cgpa_match = re.search(r"(\d\.\d{1,2})\s*(?:CGPA|GPA)", text, re.I)
    year_match = re.search(r"(?:Graduation|Expected|20)\s*(20\d{2})", text)
    
    # Skills extraction (expanded list)
    skill_keywords = [
        "Python", "Java", "C++", "JavaScript", "SQL", "React", "Node.js", "Angular",
        "Machine Learning", "AI", "Artificial Intelligence", "Data Science", "Deep Learning",
        "AWS", "Azure", "Cloud", "Docker", "Kubernetes",
        "Data Structures", "Algorithms", "System Design", "API",
        "HTML", "CSS", "MongoDB", "PostgreSQL", "MySQL",
        "Testing", "Selenium", "Jenkins", "Git", "GitHub",
        "Android", "iOS", "Flutter", "React Native", "Kotlin", "Swift"
    ]
    
    found_skills = []
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.I):
            found_skills.append(skill)
    
    # Extract values
    student_name = name_match.group(1).strip() if name_match else "Unknown"
    extracted_email = email_match.group(0) if email_match else student_email
    student_cgpa = float(cgpa_match.group(1)) if cgpa_match else 7.0
    grad_year = int(year_match.group(1)) if year_match else 2025
    student_skills = ",".join(set(found_skills)) if found_skills else "General"
    
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO students (email, name, cgpa, graduation_year, skills)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET 
            name=excluded.name,
            cgpa=excluded.cgpa,
            graduation_year=excluded.graduation_year,
            skills=excluded.skills
    """, (student_email, student_name, student_cgpa, grad_year, student_skills))
    
    conn.commit()
    conn.close()
    
    return {
        "name": student_name,
        "email": extracted_email,
        "cgpa": student_cgpa,
        "graduation_year": grad_year,
        "skills": student_skills
    }


# -------------------- Match Companies (Rule-Based - Fallback) --------------------
def match_companies(student_email, use_ml=True):
    """
    Match student profile with suitable companies.
    
    Args:
        student_email: Student's email
        use_ml: If True, use ML models; if False, use rule-based approach
    """
    if use_ml:
        try:
            from ml_company_matcher import ml_recommend_companies
            return ml_recommend_companies(student_email, model_type="random_forest")
        except Exception as e:
            print(f"ML recommendation failed: {e}. Falling back to rule-based.")
    
    # Fallback to rule-based approach
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get student details
    c.execute("SELECT * FROM students WHERE email=?", (student_email,))
    student = c.fetchone()
    
    if not student:
        conn.close()
        return []
    
    _, email, name, cgpa, grad_year, skills = student
    student_skills = set([s.strip().lower() for s in (skills or "").split(",")])
    
    # Get all companies
    c.execute("SELECT * FROM companies WHERE graduation_year=?", (grad_year,))
    companies = c.fetchall()
    conn.close()
    
    eligible_companies = []
    
    for comp in companies:
        comp_id, cname, min_cgpa, req_skills, comp_year, role, package, location, industry = comp
        required_skills = set([s.strip().lower() for s in (req_skills or "").split(",")])
        
        # Calculate skill match percentage
        matched_skills = student_skills.intersection(required_skills)
        skill_match_percent = (len(matched_skills) / len(required_skills) * 100) if required_skills else 0
        
        # Check eligibility
        is_eligible = cgpa >= min_cgpa and skill_match_percent >= 30
        
        if is_eligible:
            eligible_companies.append({
                "id": comp_id,
                "name": cname,
                "role": role,
                "package": package,
                "location": location,
                "industry": industry,
                "min_cgpa": min_cgpa,
                "required_skills": req_skills,
                "skill_match": round(skill_match_percent, 1),
                "matched_skills": list(matched_skills),
                "recommendation_type": "rule_based"
            })
    
    # Sort by skill match percentage
    eligible_companies.sort(key=lambda x: x.get('skill_match', 0), reverse=True)
    
    return eligible_companies


# -------------------- Set Selected Company --------------------
def set_selected_company(student_email, company_id):
    """Set the selected company for interview preparation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create selected_companies table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS selected_companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_id INTEGER NOT NULL,
            selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    # Ensure legacy tables have required columns
    c.execute("PRAGMA table_info(selected_companies)")
    cols = [row[1] for row in c.fetchall()]
    if "student_email" not in cols:
        try:
            c.execute("ALTER TABLE selected_companies ADD COLUMN student_email TEXT")
        except sqlite3.OperationalError:
            pass
    if "company_id" not in cols:
        try:
            c.execute("ALTER TABLE selected_companies ADD COLUMN company_id INTEGER NOT NULL")
        except sqlite3.OperationalError:
            pass
    if "selected_at" not in cols:
        try:
            c.execute("ALTER TABLE selected_companies ADD COLUMN selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass
    
    # Ensure student exists and get id
    c.execute("INSERT OR IGNORE INTO students (email) VALUES (?)", (student_email,))
    c.execute("SELECT id FROM students WHERE email = ?", (student_email,))
    student_row = c.fetchone()
    student_id = student_row[0] if student_row else None

    # Insert selection using compatible schema
    if "user_id" in cols and student_id is not None:
        # Prefer inserting user_id when present
        if "student_email" in cols:
            c.execute(
                """
                INSERT INTO selected_companies (user_id, company_id, student_email)
                VALUES (?, ?, ?)
                """,
                (student_id, company_id, student_email),
            )
        else:
            c.execute(
                """
                INSERT INTO selected_companies (user_id, company_id)
                VALUES (?, ?)
                """,
                (student_id, company_id),
            )
    else:
        c.execute(
            """
            INSERT INTO selected_companies (student_email, company_id)
            VALUES (?, ?)
            """,
            (student_email, company_id),
        )
    
    conn.commit()
    conn.close()


# -------------------- Get Selected Company --------------------
def get_selected_company(student_email):
    """Get the student's selected company."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure table exists and has schema
    c.execute("""
        CREATE TABLE IF NOT EXISTS selected_companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_id INTEGER NOT NULL,
            selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    c.execute("PRAGMA table_info(selected_companies)")
    cols = [row[1] for row in c.fetchall()]
    if "student_email" not in cols:
        try:
            c.execute("ALTER TABLE selected_companies ADD COLUMN student_email TEXT")
        except sqlite3.OperationalError:
            pass
    if "company_id" not in cols:
        try:
            c.execute("ALTER TABLE selected_companies ADD COLUMN company_id INTEGER NOT NULL")
        except sqlite3.OperationalError:
            pass
    if "selected_at" not in cols:
        try:
            c.execute("ALTER TABLE selected_companies ADD COLUMN selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass
    
    # Build query depending on schema
    c.execute("PRAGMA table_info(selected_companies)")
    cols = [row[1] for row in c.fetchall()]
    if "student_email" in cols:
        c.execute(
            """
            SELECT c.* FROM companies c
            JOIN selected_companies sc ON c.id = sc.company_id
            WHERE sc.student_email = ?
            ORDER BY sc.selected_at DESC
            LIMIT 1
            """,
            (student_email,),
        )
    elif "user_id" in cols:
        c.execute(
            """
            SELECT c.* FROM companies c
            JOIN selected_companies sc ON c.id = sc.company_id
            JOIN students s ON sc.user_id = s.id
            WHERE s.email = ?
            ORDER BY sc.selected_at DESC
            LIMIT 1
            """,
            (student_email,),
        )
    else:
        conn.close()
        return None
    
    company = c.fetchone()
    conn.close()
    
    if company:
        return {
            "id": company[0],
            "name": company[1],
            "min_cgpa": company[2],
            "required_skills": company[3],
            "graduation_year": company[4],
            "role": company[5],
            "package": company[6],
            "location": company[7],
            "industry": company[8]
        }
    return None