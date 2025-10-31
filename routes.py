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
    
    from utils import set_selected_company, get_company_name_by_id

    email = session.get("email")
    set_selected_company(email, company_id)

    # Store the selected company name in the session
    company_name = get_company_name_by_id(company_id)
    session['selected_company'] = company_name.lower().replace(" ", "_")

    flash(f"Company '{company_name}' selected! Aptitude questions will be customized for this company.", "success")
    return redirect(url_for("routes.aptitude"))

# ---- Add near your other routes in routes.py ----
import json
from flask import jsonify

@routes_bp.route("/aptitude/get-questions/<round_name>")
def api_get_round_questions(round_name):
    """
    Returns JSON payload used by aptitude.html front-end fetch.
    round_name is 'aptitude' (or could be 'technical' depending on front-end use).
    """
    if "email" not in session:
        return jsonify({"success": False, "error": "not_authenticated"}), 401

    email = session.get("email")
    # Try session first, then DB selected company
    company = session.get("selected_company")
    if not company:
        from utils import get_selected_company
        comp = get_selected_company(email)
        if comp:
            company = comp.get("name")
        else:
            return jsonify({"success": False, "error": "no_company_selected"}), 400

    # load round items
    from utils import load_company_round
    # Map front-end 'aptitude' round to dataset round
    round_key = round_name.lower()
    # choose limits: aptitude front-end requests 30 out of 100
    limit_map = {
        "aptitude": 30,
        "technical": 10,
        "gd": 1,   # for GD front-end you might fetch one topic at a time
        "hr": 20
    }
    limit = limit_map.get(round_key, None)

    items = load_company_round(company, round_key, limit=limit)

    # Do some compatibility formatting for the front-end:
    # front-end expects: { success: true, company: "TCS", questions: [{id, question, options, correct_answer, category, difficulty}, ...] }
    payload_items = []
    for it in items:
        # For aptitude items that are simple (no options) we still include options as empty list
        q = {
            "id": it.get("id"),
            "question": it.get("question") or it.get("topic") or "",
            "options": it.get("options") or [],
            "correct_answer": it.get("answer") or it.get("correct_answer") or None,
            "category": it.get("category") or "General",
            "difficulty": it.get("difficulty") or "medium",
            "time_limit": it.get("time_limit") or None,
            "explanation": it.get("explanation") or None
        }
        # If options are missing, supply empty list (frontend must handle open-ended)
        payload_items.append(q)

    return jsonify({
        "success": True,
        "company": company,
        "questions": payload_items
    })
# ---- end API endpoint ----


# ---------------------------------------------
# Aptitude Test
# ---------------------------------------------
@routes_bp.route("/aptitude")
def aptitude():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))

    # Prefer session-stored company name; fallback to DB last selection
    company = session.get("selected_company")
    if not company:
        from utils import get_selected_company
        comp = get_selected_company(session.get("email"))
        company = comp.get("name") if comp else None

    if not company:
        flash("Please select a company first!", "warning")
        return redirect(url_for("routes.companies"))

    # Render template — the front-end JS will fetch questions from /aptitude/get-questions/aptitude
    return render_template(
        "aptitude.html",
        username=session.get("name"),
        company=company.title() if company else None
    )


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
# Technical Round (NEW)
# ---------------------------------------------
@routes_bp.route("/technical")
def technical():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))

    company = session.get("selected_company")
    if not company:
        from utils import get_selected_company
        comp = get_selected_company(session.get("email"))
        company = comp.get("name") if comp else None

    if not company:
        flash("Please select a company first!", "warning")
        return redirect(url_for("routes.companies"))

    from utils import load_company_round
    questions = load_company_round(company, "technical", limit=10)

    return render_template("technical.html", username=session.get("name"), company=company.title(), questions=questions)


@routes_bp.route("/technical", endpoint="technical_round")
def technical():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    return render_template("technical.html", username=session.get("name"))


@routes_bp.route("/gd", endpoint="gd_round")
def gd():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    return render_template("gd.html", username=session.get("name"))

@routes_bp.route("/hr_interview", endpoint="hr_round")
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
        {"round": "Technical", "score": 92},
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