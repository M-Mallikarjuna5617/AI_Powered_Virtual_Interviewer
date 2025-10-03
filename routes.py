from flask import Blueprint, render_template, session, redirect, url_for, flash, request
import sqlite3
import os

# Database utility (assumes you have a get_user function)
from database import get_user

routes_bp = Blueprint("routes", __name__)

DB_PATH = "users.db"

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
    return render_template("dashboard.html", username=session.get("name"), email=session.get("email"))


# ---------------------------------------------
# Upload Documents
# ---------------------------------------------
@routes_bp.route("/upload", methods=["GET", "POST"])
def upload():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        file = request.files.get("document")
        if file:
            save_path = os.path.join("uploads", file.filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            flash("Document uploaded successfully!", "success")
        else:
            flash("No file selected!", "warning")
    return render_template("upload.html", username=session.get("name"))


# ---------------------------------------------
# Company Recommendations
# ---------------------------------------------
@routes_bp.route("/companies")
def companies():
    if "email" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("auth.login"))
    # Placeholder: Replace with real recommendation logic
    recommended_companies = ["Google", "Microsoft", "Amazon", "Tesla"]
    return render_template("companies.html", companies=recommended_companies)


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