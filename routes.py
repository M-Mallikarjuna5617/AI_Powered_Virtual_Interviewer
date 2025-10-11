from flask import Blueprint, render_template, session, redirect, url_for, flash, request, send_file
import sqlite3
import os
from werkzeug.utils import secure_filename

# Database utility
from database import get_user, save_resume, get_resume_by_email

routes_bp = Blueprint("routes", __name__)

DB_PATH = "users.db"
UPLOAD_FOLDER = "uploads/resumes"
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------
# Public / Index Routes
# ---------------------------------------------
@routes_bp.route("/")
def index():
    return render_template("index.html")

# ---------------------------------------------
# Dashboard Route
# ---------------------------------------------
@routes_bp.route("/dashboard")
def dashboard():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    
    # Check if user has uploaded a resume
    email = session.get("email")
    resume_path = get_resume_by_email(email)
    
    return render_template("dashboard.html", 
                         username=session.get("name"), 
                         email=session.get("email"),
                         has_resume=resume_path is not None,
                         resume_filename=os.path.basename(resume_path) if resume_path else None)


# ---------------------------------------------
# View Resume Route
# ---------------------------------------------
@routes_bp.route("/view-resume")
def view_resume():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    
    email = session.get("email")
    resume_path = get_resume_by_email(email)
    
    if not resume_path or not os.path.exists(resume_path):
        flash("No resume found. Please upload one first.", "warning")
        return redirect(url_for("routes.upload"))
    
    return send_file(resume_path, as_attachment=False)


# ---------------------------------------------
# Upload Documents - FIXED VERSION
# ---------------------------------------------
@routes_bp.route("/upload", methods=["GET", "POST"])
def upload():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))

    email = session["email"]
    
    if request.method == "POST":
        # Check if file was uploaded
        if 'document' not in request.files:
            flash("No file part in the request!", "danger")
            return redirect(request.url)
        
        file = request.files['document']
        
        # Check if user selected a file
        if file.filename == '':
            flash("No file selected!", "warning")
            return redirect(request.url)
        
        # Validate file type
        if file and allowed_file(file.filename):
            # Secure the filename and create unique name
            filename = secure_filename(file.filename)
            unique_filename = f"{email}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            try:
                # Save the file
                file.save(file_path)
                
                # Save to database
                save_resume(email, file_path)
                
                # Parse resume immediately after upload
                from utils import parse_resume
                try:
                    parsed_data = parse_resume(file_path, email)
                    flash(f"Resume uploaded and analyzed successfully! Found {len(parsed_data['skills'].split(','))} skills.", "success")
                except Exception as e:
                    flash(f"Resume uploaded but parsing failed: {str(e)}", "warning")
                
                return redirect(url_for("routes.companies"))
            
            except Exception as e:
                flash(f"Error uploading file: {str(e)}", "danger")
                return redirect(request.url)
        else:
            flash("Invalid file type! Please upload PDF, DOC, or DOCX only.", "danger")
            return redirect(request.url)
    
    # GET request - show upload form
    return render_template("upload.html", username=session.get("name"))


# ---------------------------------------------
# Company Recommendations
# ---------------------------------------------
@routes_bp.route("/companies")
def companies():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    
    from utils import match_companies, init_company_data, parse_resume, get_selected_company
    
    email = session.get("email")
    resume_path = get_resume_by_email(email)
    
    # Initialize company data if not exists
    init_company_data()
    
    # Check if resume is uploaded
    if not resume_path:
        flash("Please upload your resume first to get company recommendations!", "warning")
        return redirect(url_for("routes.upload"))
    
    # Parse resume if not already parsed
    try:
        parse_resume(resume_path, email)
    except Exception as e:
        flash(f"Error parsing resume: {str(e)}", "danger")
    
    # Get matched companies
    recommended_companies = match_companies(email)
    selected_company = get_selected_company(email)
    
    return render_template("companies.html", 
                         companies=recommended_companies,
                         selected_company=selected_company,
                         username=session.get("name"))


@routes_bp.route("/select_company/<int:company_id>", methods=["POST"])
def select_company(company_id):
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    
    from utils import set_selected_company
    
    email = session.get("email")
    set_selected_company(email, company_id)
    
    flash("Company selected! Interview questions will be customized for this company.", "success")
    return redirect(url_for("routes.companies"))


# ---------------------------------------------
# Aptitude Test
# ---------------------------------------------
@routes_bp.route("/aptitude")
def aptitude():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    return render_template("aptitude.html", username=session.get("name"))


# ---------------------------------------------
# Group Discussion Simulation
# ---------------------------------------------
@routes_bp.route("/gd")
def gd():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    return render_template("gd.html", username=session.get("name"))


# ---------------------------------------------
# AI HR Interview
# ---------------------------------------------
@routes_bp.route("/hr_interview")
def hr_interview():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    return render_template("hr.html", username=session.get("name"))


# ---------------------------------------------
# Feedback & Reports
# ---------------------------------------------
@routes_bp.route("/feedback")
def feedback():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    # Placeholder: Replace with real feedback logic
    feedback_list = [
        {"round": "Aptitude", "score": 85},
        {"round": "GD", "score": 90},
        {"round": "HR", "score": 88},
    ]
    return render_template("feedback.html", feedback=feedback_list)


# ---------------------------------------------
# Settings Page
# ---------------------------------------------
@routes_bp.route("/settings", methods=["GET", "POST"])
def settings():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        # Example: update user settings
        new_name = request.form.get("name")
        if new_name:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("UPDATE users SET name=? WHERE email=?", (new_name, session["email"]))
            conn.commit()
            conn.close()
            session["name"] = new_name
            flash("Settings updated successfully!", "success")
    return render_template("settings.html", username=session.get("name"))