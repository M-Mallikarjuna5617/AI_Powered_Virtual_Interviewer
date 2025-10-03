CREATE TABLE companies (
    id INT IDENTITY(1,1) PRIMARY KEY,  -- COMMA added here
    name VARCHAR(100) NOT NULL,
    min_cgpa DECIMAL(3,2) DEFAULT 0.00,
    required_skills VARCHAR(255),      -- comma-separated (e.g., 'Python,Java,SQL')
    graduation_year INT,                -- eligible passout year
    role VARCHAR(100),
    package_offered VARCHAR(50),
    location VARCHAR(100)
);
