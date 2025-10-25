"""
Quick Start Script for AI-Powered Virtual Interviewer
Simple setup without Unicode characters for Windows compatibility
"""

import os
import sys
import subprocess
import sqlite3

def main():
    print("\n" + "="*80)
    print("AI-POWERED VIRTUAL INTERVIEWER - QUICK START")
    print("="*80)
    print("Setting up comprehensive interview simulation system")
    print("="*80 + "\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        return False
    
    print("Python version: OK")
    
    # Install basic dependencies
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False
    
    # Initialize database
    print("\nInitializing database...")
    try:
        from database import init_db
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False
    
    # Create directories
    print("\nCreating directories...")
    directories = ['uploads', 'uploads/resumes', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    # Initialize question banks
    print("\nInitializing question banks...")
    try:
        from question_generator import initialize_question_bank
        initialize_question_bank()
        print("Aptitude questions initialized")
        
        from technical import init_technical_questions
        init_technical_questions()
        print("Technical questions initialized")
        
        from gd import init_gd_topics
        init_gd_topics()
        print("GD topics initialized")
        
        from hr import init_hr_questions
        init_hr_questions()
        print("HR questions initialized")
        
    except Exception as e:
        print(f"Question bank initialization failed: {e}")
        return False
    
    # Train ML model
    print("\nTraining ML model...")
    try:
        from ml_company_matcher import CompanyRecommendationEngine
        engine = CompanyRecommendationEngine()
        accuracy = engine.train_model()
        print(f"ML model trained with {accuracy:.2f} accuracy")
    except Exception as e:
        print(f"ML model training failed: {e}")
        return False
    
    # Test system
    print("\nTesting system...")
    try:
        # Test database
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM companies")
        company_count = c.fetchone()[0]
        conn.close()
        print(f"Database test: {company_count} companies found")
        
        # Test ML recommendations
        from ml_company_matcher import ml_recommend_companies
        recommendations = ml_recommend_companies("test@example.com")
        print(f"ML test: {len(recommendations)} recommendations generated")
        
        print("System test: PASSED")
        
    except Exception as e:
        print(f"System test failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update .env file with your API keys")
    print("2. Run: python app.py")
    print("3. Open: http://localhost:5000")
    print("\nYour AI-Powered Virtual Interviewer is ready!")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)