CREATE TABLE students (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    cgpa DECIMAL(3,2),
    graduation_year INT,
    skills VARCHAR(255),
    selected_company_id INT,
    selected_company_name VARCHAR(100)
);

CREATE TABLE resumes (
    id INT IDENTITY(1,1) PRIMARY KEY
    student_id INT,
    file_path VARCHAR(255)
);

CREATE TABLE companies (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    min_cgpa DECIMAL(3,2) DEFAULT 0.00,
    required_skills VARCHAR(255),
    graduation_year INT,
    role VARCHAR(100),
    package_offered VARCHAR(50),
    location VARCHAR(100)
);
