import mysql.connector

# === DB Connection ===
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yourpassword",
    database="virtual_interviewer"
)
cursor = db.cursor(dictionary=True)

def match_companies(student_id=1):
    # Get student info
    cursor.execute("SELECT * FROM students WHERE id=%s", (student_id,))
    student = cursor.fetchone()
    if not student:
        print("❌ No student found")
        return []

    student_skills = set([s.strip().lower() for s in student['skills'].split(",")])
    cgpa = student['cgpa']
    grad_year = student['graduation_year']

    # Get companies
    cursor.execute("SELECT * FROM companies")
    companies = cursor.fetchall()

    eligible = []
    for comp in companies:
        required_skills = set([s.strip().lower() for s in comp['required_skills'].split(",")])

        if (cgpa >= comp['min_cgpa'] and
            grad_year == comp['graduation_year'] and
            required_skills.issubset(student_skills)):

            eligible.append(comp)

    print("✅ Eligible Companies for", student['name'])
    for c in eligible:
        print(f"- {c['name']} | Role: {c['role']} | Package: {c['package_offered']} | Location: {c['location']}")

    return eligible

# Example usage:
# match_companies(student_id=1)
