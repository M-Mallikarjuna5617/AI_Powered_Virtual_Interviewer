
"""
#!/usr/bin/env python3
\"\"\"
Complete Setup Script for API Project
Run this to initialize everything
\"\"\"

import subprocess
import sys
import os

def run_command(command, description):
    \"\"\"Execute a command and show progress\"\"\"
    print(f"\\n{'='*60}")
    print(f"🔧 {description}")
    print('='*60)
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - Complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        return False

def setup_project():
    \"\"\"Main setup function\"\"\"
    print("\\n" + "="*60)
    print("🚀 AI-Powered Virtual Interviewer Setup")
    print("="*60)
    
    steps = [
        ("pip install -r requirements.txt", "Installing Python dependencies"),
        ("python -m spacy download en_core_web_sm", "Downloading SpaCy language model"),
        ("python -c 'from database import init_db; init_db()'", "Initializing database"),
        ("python fix_company_db.py", "Setting up company data"),
        ("python enhanced_question_generator.py", "Populating question banks"),
        ("python -c 'from aptitude_routes import init_aptitude_tables; init_aptitude_tables()'", "Creating aptitude tables"),
    ]
    
    success_count = 0
    for command, description in steps:
        if run_command(command, description):
            success_count += 1
    
    print("\\n" + "="*60)
    print(f"📊 Setup Summary: {success_count}/{len(steps)} steps completed")
    print("="*60)
    
    if success_count == len(steps):
        print("\\n✅ Setup complete! Next steps:")
        print("\\n1. Create .env file with your API keys:")
        print("   - OpenAI API Key")
        print("   - Judge0 API Key (from RapidAPI)")
        print("   - Google OAuth credentials")
        print("\\n2. Run the application:")
        print("   python app.py")
        print("\\n3. Open browser:")
        print("   http://localhost:5000")
    else:
        print("\\n⚠️  Some steps failed. Please check errors above.")

if __name__ == "__main__":
    setup_project()
"""
