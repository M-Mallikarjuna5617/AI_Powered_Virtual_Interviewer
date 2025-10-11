import sqlite3

DB_PATH = "users.db"

def fix_company_database():
    """Add industry column to companies table and initialize data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if industry column exists
    cursor.execute("PRAGMA table_info(companies)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'industry' not in columns:
        print("Adding 'industry' column to companies table...")
        cursor.execute("ALTER TABLE companies ADD COLUMN industry TEXT")
        conn.commit()
        print("✅ Industry column added!")
    else:
        print("✅ Industry column already exists!")
    
    # Create selected_companies table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS selected_companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_email TEXT NOT NULL,
            company_id INTEGER NOT NULL,
            selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    """)
    print("✅ Selected companies table created/verified!")
    
    # Check if companies table is empty
    cursor.execute("SELECT COUNT(*) FROM companies")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("Initializing company data...")
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
        
        cursor.executemany("""
            INSERT INTO companies (name, min_cgpa, required_skills, graduation_year, role, package_offered, location, industry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, companies_data)
        
        conn.commit()
        print(f"✅ {len(companies_data)} companies added to database!")
    else:
        print(f"✅ Companies table already has {count} companies!")
    
    conn.close()
    print("\n🎉 Database update complete!")

if __name__ == "__main__":
    fix_company_database()