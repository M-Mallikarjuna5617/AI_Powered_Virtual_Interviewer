
"""
Create sample students and test data for testing
"""

import sqlite3
from datetime import datetime

DB_PATH = "users.db"

def create_sample_students():
    """Create sample student accounts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    sample_students = [
        {
            "email": "test1@example.com",
            "name": "Rahul Kumar",
            "cgpa": 8.5,
            "graduation_year": 2025,
            "skills": "Python,Java,Machine Learning,SQL,Data Structures"
        },
        {
            "email": "test2@example.com",
            "name": "Priya Sharma",
            "cgpa": 7.8,
            "graduation_year": 2025,
            "skills": "Java,Spring Boot,MySQL,AWS,Docker"
        },
        {
            "email": "test3@example.com",
            "name": "Amit Patel",
            "cgpa": 9.2,
            "graduation_year": 2025,
            "skills": "Python,TensorFlow,Deep Learning,OpenCV,NLP"
        }
    ]
    
    for student in sample_students:
        c.execute("""
            INSERT OR REPLACE INTO students (email, name, cgpa, graduation_year, skills)
            VALUES (?, ?, ?, ?, ?)
        """, (student["email"], student["name"], student["cgpa"], 
              student["graduation_year"], student["skills"]))
        
        # Create user account
        c.execute("""
            INSERT OR IGNORE INTO users (name, email, password, provider)
            VALUES (?, ?, 'test123', 'email')
        """, (student["name"], student["email"]))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created {len(sample_students)} sample students")
    for s in sample_students:
        print(f"   📧 {s['email']} (password: test123)")
