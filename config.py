# Production Configuration
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
