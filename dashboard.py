from flask import Blueprint, render_template, request, redirect, url_for, session, send_file, flash
import os
from database import save_resume, get_resume_by_email

dashboard_bp = Blueprint("dashboard", __name__)
UPLOAD_FOLDER = "uploads/resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@dashboard_bp.route("/upload_resume", methods=["GET", "POST"])
def upload_resume():
    if "email" not in session:
        flash("Please log in first.", "danger")
        return redirect(url_for("auth.login"))

    email = session["email"]
    resume_path = get_resume_by_email(email)

    if request.method == "POST":
        file = request.files.get("resume")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        if not file.filename.lower().endswith(".pdf"):
            flash("Only PDF files are allowed.", "danger")
            return redirect(request.url)

        # Save file
        filename = f"{email}_resume.pdf"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Save path to DB
        save_resume(email, file_path)
        flash("Resume uploaded successfully!", "success")
        return redirect(url_for("dashboard.upload_resume"))

    return render_template("upload.html", resume=resume_path)


@dashboard_bp.route("/view_resume/<email>")
def view_resume(email):
    resume_path = get_resume_by_email(email)
    if not resume_path or not os.path.exists(resume_path):
        flash("No resume found for this user.", "danger")
        return redirect(url_for("dashboard.upload_resume"))
    return send_file(resume_path, mimetype="application/pdf")
