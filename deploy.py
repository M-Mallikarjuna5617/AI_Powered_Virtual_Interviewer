"""
Deployment Script for AI-Powered Virtual Interviewer
Handles production deployment and configuration
"""

import os
import sys
import subprocess
import sqlite3
from datetime import datetime

def print_deployment_banner():
    """Print deployment banner"""
    print("\n" + "="*80)
    print("🚀 AI-POWERED VIRTUAL INTERVIEWER - DEPLOYMENT")
    print("="*80)
    print("📦 Deploying comprehensive interview simulation system")
    print("🌐 Production-ready configuration")
    print("="*80 + "\n")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 2 * 1024**3:  # 2GB
            print("⚠️  Warning: Less than 2GB RAM available")
        else:
            print(f"✅ Memory: {memory.total // (1024**3)}GB available")
    except ImportError:
        print("ℹ️  psutil not available, skipping memory check")
    
    return True

def install_production_dependencies():
    """Install production dependencies"""
    print("\n📦 Installing production dependencies...")
    
    production_packages = [
        "gunicorn==21.2.0",
        "gevent==23.7.0",
        "supervisor==4.2.5",
        "nginx-config==0.0.1"
    ]
    
    for package in production_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {package}")
            return False
    
    return True

def configure_production_environment():
    """Configure production environment"""
    print("\n🔧 Configuring production environment...")
    
    # Create production config
    config_content = """# Production Configuration
import os

class ProductionConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'change_this_in_production')
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    JUDGE0_API_KEY = os.environ.get('JUDGE0_API_KEY')
    
    # Flask settings
    DEBUG = False
    TESTING = False
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("✅ Production configuration created")
    
    # Create environment file
    env_content = """# Production Environment Variables
SECRET_KEY=your_super_secret_production_key_here
DATABASE_URL=sqlite:///users.db
OPENAI_API_KEY=your_openai_api_key_here
JUDGE0_API_KEY=your_judge0_api_key_here

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False

# Server Configuration
HOST=0.0.0.0
PORT=5000
WORKERS=4
"""
    
    with open('.env.production', 'w') as f:
        f.write(env_content)
    print("✅ Production environment file created")
    
    return True

def create_gunicorn_config():
    """Create Gunicorn configuration"""
    print("\n🦄 Creating Gunicorn configuration...")
    
    gunicorn_config = """# Gunicorn Configuration for AI-Powered Virtual Interviewer
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'ai_interviewer'

# Server mechanics
daemon = False
pidfile = 'logs/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment if using HTTPS)
# keyfile = 'ssl/private.key'
# certfile = 'ssl/certificate.crt'
"""
    
    with open('gunicorn.conf.py', 'w') as f:
        f.write(gunicorn_config)
    print("✅ Gunicorn configuration created")
    
    return True

def create_supervisor_config():
    """Create Supervisor configuration"""
    print("\n👨‍💼 Creating Supervisor configuration...")
    
    supervisor_config = """[program:ai_interviewer]
command=/path/to/venv/bin/gunicorn --config gunicorn.conf.py app:app
directory=/path/to/AI_Powered_Virtual_Interviewer
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/path/to/AI_Powered_Virtual_Interviewer/logs/supervisor.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
"""
    
    with open('ai_interviewer.conf', 'w') as f:
        f.write(supervisor_config)
    print("✅ Supervisor configuration created")
    
    return True

def create_nginx_config():
    """Create Nginx configuration"""
    print("\n🌐 Creating Nginx configuration...")
    
    nginx_config = """server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /path/to/ssl/certificate.crt;
    ssl_certificate_key /path/to/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Static files
    location /static {
        alias /path/to/AI_Powered_Virtual_Interviewer/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Upload files
    location /uploads {
        alias /path/to/AI_Powered_Virtual_Interviewer/uploads;
        expires 1d;
    }
    
    # Main application
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # File upload size limit
    client_max_body_size 20M;
}
"""
    
    with open('nginx.conf', 'w') as f:
        f.write(nginx_config)
    print("✅ Nginx configuration created")
    
    return True

def create_deployment_scripts():
    """Create deployment scripts"""
    print("\n📜 Creating deployment scripts...")
    
    # Start script
    start_script = """#!/bin/bash
# Start AI-Powered Virtual Interviewer

echo "🚀 Starting AI-Powered Virtual Interviewer..."

# Activate virtual environment
source venv/bin/activate

# Start with Gunicorn
gunicorn --config gunicorn.conf.py app:app

echo "✅ Application started successfully!"
"""
    
    with open('start_production.sh', 'w') as f:
        f.write(start_script)
    os.chmod('start_production.sh', 0o755)
    print("✅ Start script created")
    
    # Stop script
    stop_script = """#!/bin/bash
# Stop AI-Powered Virtual Interviewer

echo "🛑 Stopping AI-Powered Virtual Interviewer..."

# Kill Gunicorn processes
pkill -f gunicorn

echo "✅ Application stopped successfully!"
"""
    
    with open('stop_production.sh', 'w') as f:
        f.write(stop_script)
    os.chmod('stop_production.sh', 0o755)
    print("✅ Stop script created")
    
    # Restart script
    restart_script = """#!/bin/bash
# Restart AI-Powered Virtual Interviewer

echo "🔄 Restarting AI-Powered Virtual Interviewer..."

./stop_production.sh
sleep 2
./start_production.sh

echo "✅ Application restarted successfully!"
"""
    
    with open('restart_production.sh', 'w') as f:
        f.write(restart_script)
    os.chmod('restart_production.sh', 0o755)
    print("✅ Restart script created")
    
    return True

def create_log_directories():
    """Create log directories"""
    print("\n📁 Creating log directories...")
    
    log_dirs = ['logs', 'logs/nginx', 'logs/gunicorn', 'logs/supervisor']
    
    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)
        print(f"  ✅ {log_dir}")
    
    return True

def run_system_tests():
    """Run comprehensive system tests"""
    print("\n🧪 Running system tests...")
    
    try:
        # Test database
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM companies")
        company_count = c.fetchone()[0]
        conn.close()
        print(f"  ✅ Database: {company_count} companies")
        
        # Test ML model
        from ml_company_matcher import ml_recommend_companies
        recommendations = ml_recommend_companies("test@example.com")
        print(f"  ✅ ML Engine: {len(recommendations)} recommendations")
        
        # Test AI services
        from ai_service import CodeExecutionService, HRInterviewService
        code_service = CodeExecutionService()
        hr_service = HRInterviewService()
        print("  ✅ AI Services: Initialized")
        
        print("✅ All system tests passed!")
        return True
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def create_monitoring_setup():
    """Create monitoring setup"""
    print("\n📊 Creating monitoring setup...")
    
    monitoring_script = """#!/bin/bash
# Monitoring script for AI-Powered Virtual Interviewer

echo "📊 System Monitoring Report"
echo "=========================="

# Check if application is running
if pgrep -f gunicorn > /dev/null; then
    echo "✅ Application: Running"
else
    echo "❌ Application: Not running"
fi

# Check database
if [ -f "users.db" ]; then
    echo "✅ Database: Present"
    echo "   Size: $(du -h users.db | cut -f1)"
else
    echo "❌ Database: Missing"
fi

# Check logs
if [ -f "logs/error.log" ]; then
    echo "✅ Error Log: Present"
    echo "   Recent errors: $(tail -5 logs/error.log | wc -l)"
else
    echo "ℹ️  Error Log: Not found"
fi

# Check disk space
echo "💾 Disk Usage:"
df -h | grep -E "(Filesystem|/dev/)"

# Check memory usage
echo "🧠 Memory Usage:"
free -h

echo "=========================="
"""
    
    with open('monitor.sh', 'w') as f:
        f.write(monitoring_script)
    os.chmod('monitor.sh', 0o755)
    print("✅ Monitoring script created")
    
    return True

def main():
    """Main deployment function"""
    print_deployment_banner()
    
    deployment_steps = [
        ("Checking system requirements", check_system_requirements),
        ("Installing production dependencies", install_production_dependencies),
        ("Configuring production environment", configure_production_environment),
        ("Creating Gunicorn configuration", create_gunicorn_config),
        ("Creating Supervisor configuration", create_supervisor_config),
        ("Creating Nginx configuration", create_nginx_config),
        ("Creating deployment scripts", create_deployment_scripts),
        ("Creating log directories", create_log_directories),
        ("Creating monitoring setup", create_monitoring_setup),
        ("Running system tests", run_system_tests)
    ]
    
    success_count = 0
    total_steps = len(deployment_steps)
    
    for step_name, step_function in deployment_steps:
        print(f"\n🔄 {step_name}...")
        if step_function():
            success_count += 1
            print(f"✅ {step_name} completed!")
        else:
            print(f"❌ {step_name} failed!")
    
    print("\n" + "="*80)
    print("📊 DEPLOYMENT SUMMARY")
    print("="*80)
    print(f"✅ Successful steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("\n🎉 PRODUCTION DEPLOYMENT READY!")
        print("\n📋 Next steps:")
        print("1. Update configuration files with your domain and paths")
        print("2. Set up SSL certificates")
        print("3. Configure Nginx and Supervisor")
        print("4. Run: ./start_production.sh")
        print("5. Monitor with: ./monitor.sh")
        print("\n🚀 Your AI-Powered Virtual Interviewer is production-ready!")
    else:
        print(f"\n⚠️  Deployment completed with {total_steps - success_count} issues")
        print("Please check the errors above and run deployment again")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()