"""
Complete System Setup Script
Initializes all components of the AI-Powered Virtual Interviewer
"""

import os
import sqlite3
import subprocess
import sys
from datetime import datetime

def print_banner():
    """Print setup banner"""
    print("\n" + "="*80)
    print("AI-POWERED VIRTUAL INTERVIEWER - COMPLETE SYSTEM SETUP")
    print("="*80)
    print("Setting up comprehensive interview simulation system")
    print("Including: Aptitude, Technical, GD, HR, and AI-powered feedback")
    print("="*80 + "\n")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'flask', 'requests', 'pdfplumber', 'reportlab', 'spacy',
        'scikit-learn', 'pandas', 'numpy', 'joblib', 'openai',
        'textblob', 'nltk', 'speechrecognition'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"  ✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"  ❌ Failed to install {package}")
                return False
    
    print("✅ All dependencies satisfied!")
    return True

def initialize_database():
    """Initialize database with all tables"""
    print("\n🗄️  Initializing database...")
    
    try:
        from database import init_db
        init_db()
        print("✅ Database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def initialize_question_banks():
    """Initialize all question banks"""
    print("\n📚 Initializing question banks...")
    
    try:
        # Initialize aptitude questions
        from question_generator import initialize_question_bank
        initialize_question_bank()
        print("  ✅ Aptitude questions initialized")
        
        # Initialize technical questions
        from technical import init_technical_questions
        init_technical_questions()
        print("  ✅ Technical questions initialized")
        
        # Initialize GD topics
        from gd import init_gd_topics
        init_gd_topics()
        print("  ✅ GD topics initialized")
        
        # Initialize HR questions
        from hr import init_hr_questions
        init_hr_questions()
        print("  ✅ HR questions initialized")
        
        print("✅ All question banks initialized!")
        return True
    except Exception as e:
        print(f"❌ Question bank initialization failed: {e}")
        return False

def initialize_ml_models():
    """Initialize ML models for company recommendation"""
    print("\n🤖 Initializing ML models...")
    
    try:
        from ml_company_matcher import CompanyRecommendationEngine
        engine = CompanyRecommendationEngine()
        accuracy = engine.train_model()
        print(f"✅ ML model trained with {accuracy:.2f} accuracy")
        return True
    except Exception as e:
        print(f"❌ ML model initialization failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'uploads',
        'uploads/resumes',
        'models',
        'static/css',
        'static/js',
        'static/images',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")
    
    print("✅ All directories created!")
    return True

def setup_environment():
    """Setup environment variables"""
    print("\n🔧 Setting up environment...")
    
    env_content = """# AI-Powered Virtual Interviewer Environment Variables
SECRET_KEY=your_super_secret_key_change_in_production
OPENAI_API_KEY=your_openai_api_key_here
JUDGE0_API_KEY=your_judge0_api_key_here

# Database
DATABASE_URL=sqlite:///users.db

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Environment file created (.env)")
    else:
        print("✅ Environment file already exists")
    
    return True

def test_system():
    """Test the complete system"""
    print("\n🧪 Testing system components...")
    
    try:
        # Test database connection
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM companies")
        company_count = c.fetchone()[0]
        conn.close()
        print(f"  ✅ Database connection: {company_count} companies found")
        
        # Test ML model
        from ml_company_matcher import ml_recommend_companies
        test_recommendations = ml_recommend_companies("test@example.com")
        print(f"  ✅ ML recommendations: {len(test_recommendations)} companies")
        
        # Test AI services
        from ai_service import CodeExecutionService, HRInterviewService
        code_service = CodeExecutionService()
        hr_service = HRInterviewService()
        print("  ✅ AI services initialized")
        
        print("✅ System test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def create_startup_script():
    """Create startup script"""
    print("\n📝 Creating startup script...")
    
    startup_content = """#!/bin/bash
# AI-Powered Virtual Interviewer Startup Script

echo "🚀 Starting AI-Powered Virtual Interviewer..."
echo "=" * 50

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Initialize database
echo "🗄️  Initializing database..."
python -c "from database import init_db; init_db()"

# Start the application
echo "🚀 Starting Flask application..."
python app.py
"""
    
    with open('start.sh', 'w') as f:
        f.write(startup_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod('start.sh', 0o755)
    
    print("✅ Startup script created (start.sh)")
    return True

def main():
    """Main setup function"""
    print_banner()
    
    setup_steps = [
        ("Checking dependencies", check_dependencies),
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Initializing database", initialize_database),
        ("Initializing question banks", initialize_question_banks),
        ("Initializing ML models", initialize_ml_models),
        ("Creating startup script", create_startup_script),
        ("Testing system", test_system)
    ]
    
    success_count = 0
    total_steps = len(setup_steps)
    
    for step_name, step_function in setup_steps:
        print(f"\n🔄 {step_name}...")
        if step_function():
            success_count += 1
            print(f"✅ {step_name} completed!")
        else:
            print(f"❌ {step_name} failed!")
    
    print("\n" + "="*80)
    print("📊 SETUP SUMMARY")
    print("="*80)
    print(f"✅ Successful steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("\n🎉 COMPLETE SYSTEM SETUP SUCCESSFUL!")
        print("\n📋 Next steps:")
        print("1. Update .env file with your API keys")
        print("2. Run: python app.py")
        print("3. Open: http://localhost:5000")
        print("\n🚀 Your AI-Powered Virtual Interviewer is ready!")
    else:
        print(f"\n⚠️  Setup completed with {total_steps - success_count} issues")
        print("Please check the errors above and run setup again")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
