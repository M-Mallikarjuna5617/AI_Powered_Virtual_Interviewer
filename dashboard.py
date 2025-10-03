from flask import Blueprint, render_template, session, redirect, url_for, flash, request
import os
from utils import parse_resume, match_companies

# Blueprint
dashboard_bp = Blueprint('dashboard', __name__, template_folder='templates', url_prefix='/dashboard')

# -------------------- Login Required Decorator --------------------
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access the dashboard.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

# -------------------- Helper to get username --------------------
def get_username():
    return session.get('user', {}).get('name', 'User')

# -------------------- Dashboard Home --------------------
@dashboard_bp.route('/')
@login_required
def index():
    return render_template('dashboard.html', username=get_username())

# -------------------- Resume Upload --------------------
@dashboard_bp.route('/upload_resume', methods=['GET', 'POST'])
@login_required
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('resume')
        if file and file.filename.endswith(('.pdf', '.docx')):
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", file.filename)
            file.save(save_path)

            # Parse resume & update student profile
            student_email = session['user']['email']
            student_data = parse_resume(save_path, student_email)

            flash(f"Resume uploaded & profile updated for {student_data['name']}", "success")
            return redirect(url_for('dashboard.upload_resume'))
        else:
            flash("Please upload a valid PDF or DOCX file.", "warning")

    return render_template('upload.html', username=get_username())

# -------------------- Companies --------------------
@dashboard_bp.route('/companies')
@login_required
def companies_page():
    student_email = session['user']['email']
    eligible_companies = match_companies(student_email)
    return render_template('companies.html', username=get_username(), companies=eligible_companies)

# -------------------- Group Discussion --------------------
@dashboard_bp.route('/gd')
@login_required
def gd():
    return render_template('gd.html', username=get_username())

# -------------------- HR Interview --------------------
@dashboard_bp.route('/hr-interview')
@login_required
def hr_interview():
    return render_template('hr_interview.html', username=get_username())

# -------------------- Feedback --------------------
@dashboard_bp.route('/feedback')
@login_required
def feedback():
    reports = [
        {"module": "Aptitude Test", "score": "85%", "status": "Pass", "feedback": "Good performance, focus on speed."},
        {"module": "Group Discussion", "score": "78%", "status": "Pass", "feedback": "Participate more actively."},
        {"module": "HR Interview", "score": "90%", "status": "Excellent", "feedback": "Great communication skills."}
    ]
    return render_template('feedback.html', username=get_username(), reports=reports)

# -------------------- Settings --------------------
@dashboard_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user = session.get('user', {})
    if request.method == 'POST':
        # Profile update
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        profile_pic = request.files.get('profile_pic')
        if full_name: user['name'] = full_name
        if email: user['email'] = email
        if profile_pic and profile_pic.filename != '':
            user['profile_pic'] = profile_pic.filename
            flash("Profile picture updated (demo).", "success")

        # Password change
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_new_password')
        if current_password and new_password and confirm_password:
            if current_password != "password123":
                flash("Current password is incorrect.", "error")
            elif new_password != confirm_password:
                flash("New passwords do not match.", "error")
            else:
                user['password'] = new_password
                flash("Password updated successfully!", "success")

        # Notification preferences
        user['notifications'] = {
            "email": bool(request.form.get('email_notifications')),
            "sms": bool(request.form.get('sms_notifications')),
            "gd_reminders": bool(request.form.get('gd_reminders'))
        }
        session['user'] = user
        flash("Settings updated successfully!", "success")

    return render_template('settings.html', username=user.get('name', 'User'))
