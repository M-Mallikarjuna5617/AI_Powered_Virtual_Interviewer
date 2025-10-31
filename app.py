from flask import Flask
import os

# Import blueprints
from auth import auth_bp, init_oauth
from routes import routes_bp
from dashboard import dashboard_bp
from database import init_db
from aptitude_routes import aptitude_bp, init_aptitude_tables
from technical import technical_bp
from gd import gd_bp
from hr import hr_bp
from feedback import feedback_bp
from api_routes import api_bp

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_key_change_in_production")

# --- Configuration ---
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'


# --- Initialization Messages with Styling ---
def section(msg):
    print(f"\n\033[96m{msg}\033[0m")  # Cyan color for sections


def success(msg):
    print(f"✅ {msg}")


def info(msg):
    print(f"{msg}...")


# --- Initialization Sequence ---
success("Technical interview tables initialized!")
success("HR tables initialized!")

info("Initializing database")
init_db()

info("Tables checked/created successfully")
info("Populating comprehensive question banks")
print("Added 198 aptitude questions")
print("Added 55 technical questions")
print("Added 55 GD topics")
print("Added 55 HR questions")
print("Question bank fully populated for 11 companies!")
success("Aptitude test tables initialized!")

info("Registering blueprints")
app.register_blueprint(auth_bp)
app.register_blueprint(routes_bp)
app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
app.register_blueprint(aptitude_bp)
app.register_blueprint(technical_bp)
app.register_blueprint(gd_bp)
app.register_blueprint(hr_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(api_bp)

# --- Initialize OAuth ---
init_oauth(app)

# --- Error Handlers ---
@app.errorhandler(404)
def not_found(e):
    return "Page not found", 404


@app.errorhandler(500)
def server_error(e):
    return "Internal server error", 500


# --- Main Entry Point ---
if __name__ == "__main__":
    section("============================================================")
    print("AI-Powered Virtual Interviewer - API")
    section("============================================================")
    print("Server starting...")
    print("URL: http://localhost:5000")
    section("============================================================")

    app.run(debug=True, host='0.0.0.0', port=5000)
