#!/usr/bin/env python3
"""
Test script to check aptitude functionality
"""

import sqlite3
import requests
import json

# Test database connection and questions
def test_database():
    print("=== Testing Database ===")
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        # Check if question_bank table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='question_bank'")
        table_exists = c.fetchone()
        print(f"question_bank table exists: {table_exists is not None}")
        
        if table_exists:
            # Count questions
            c.execute("SELECT COUNT(*) FROM question_bank")
            count = c.fetchone()[0]
            print(f"Number of questions in question_bank: {count}")
            
            # Get sample questions
            c.execute("SELECT id, question_text, correct_answer FROM question_bank LIMIT 3")
            sample_questions = c.fetchall()
            print("Sample questions:")
            for q in sample_questions:
                print(f"  ID: {q[0]}, Question: {q[1][:50]}..., Answer: {q[2]}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

# Test aptitude endpoint
def test_aptitude_endpoint():
    print("\n=== Testing Aptitude Endpoint ===")
    try:
        # First, let's try to login to get a session
        login_data = {
            "email": "test@example.com",
            "password": "test123"
        }
        
        session = requests.Session()
        
        # Try to login using test-login endpoint
        try:
            login_response = session.post("http://127.0.0.1:5000/test-login", json=login_data)
            print(f"Login response status: {login_response.status_code}")
            if login_response.status_code == 200:
                print("Login successful!")
        except Exception as e:
            print(f"Login failed: {e}")
        
        # Test aptitude endpoint
        response = session.get("http://127.0.0.1:5000/aptitude/get-questions/aptitude")
        print(f"Aptitude endpoint status: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Endpoint error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Aptitude System...")
    
    db_ok = test_database()
    endpoint_ok = test_aptitude_endpoint()
    
    print(f"\n=== Results ===")
    print(f"Database: {'OK' if db_ok else 'FAILED'}")
    print(f"Endpoint: {'OK' if endpoint_ok else 'FAILED'}")
    
    if not db_ok:
        print("\nTo fix database issues, run: python initialize_new_tables.py")
