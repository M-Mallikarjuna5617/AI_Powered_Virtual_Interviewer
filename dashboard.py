from flask import Blueprint, render_template, session, redirect, url_for, flash, request

dashboard_bp = Blueprint('dashboard', __name__, template_folder='templates')

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

# -------------------- Feature Pages --------------------
@dashboard_bp.route('/aptitude')
@login_required
def aptitude():
    topics = [
        {"id": 1, "name": "Quantitative Aptitude", "description": "Math and problem-solving questions."},
        {"id": 2, "name": "Logical Reasoning", "description": "Assess logical thinking skills."},
        {"id": 3, "name": "Verbal Ability", "description": "English grammar and comprehension tests."},
    ]
    return render_template('aptitude.html', username=get_username(), topics=topics)

@dashboard_bp.route('/gd')
@login_required
def gd():
    return render_template('gd.html', username=get_username())

@dashboard_bp.route('/hr-interview')
@login_required
def hr_interview():
    return render_template('hr_interview.html', username=get_username())

@dashboard_bp.route('/feedback')
@login_required
def feedback():
    reports = [
        {"module": "Aptitude Test", "score": "85%", "status": "Pass", "feedback": "Good performance, focus on speed."},
        {"module": "Group Discussion", "score": "78%", "status": "Pass", "feedback": "Participate more actively."},
        {"module": "HR Interview", "score": "90%", "status": "Excellent", "feedback": "Great communication skills."}
    ]
    return render_template('feedback.html', username=get_username(), reports=reports)

# -------------------- Dropdown Pages --------------------
@dashboard_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        # Here you can implement actual file upload logic
        flash("Files uploaded successfully!", "success")
    return render_template('upload.html', username=get_username())

@dashboard_bp.route('/companies')
@login_required
def companies():
    companies = [
        {"name": "Google", "position": "Software Engineer", "description": "Tech giant, world-class projects.", "apply_link": "#"},
        {"name": "Amazon", "position": "Data Analyst", "description": "E-commerce leader, data-driven culture.", "apply_link": "#"},
        {"name": "Microsoft", "position": "Product Manager", "description": "Innovative cloud solutions.", "apply_link": "#"},
        {"name": "Infosys", "position": "Software Developer", "description": "Global IT services company.", "apply_link": "#"},
    ]
    return render_template('companies.html', username=get_username(), companies=companies)

# -------------------- Settings Page --------------------
@dashboard_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user = session.get('user', {})
    if request.method == 'POST':
        # -------- Profile Update --------
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        profile_pic = request.files.get('profile_pic')  # Save file logic goes here
        if full_name:
            user['name'] = full_name
        if email:
            user['email'] = email
        # For demo, we just store filename in session
        if profile_pic and profile_pic.filename != '':
            user['profile_pic'] = profile_pic.filename
            flash("Profile picture updated (demo).", "success")

        # -------- Password Change --------
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_new_password')
        # For demo, assume current password is "password123"
        if current_password and new_password and confirm_password:
            if current_password != "password123":
                flash("Current password is incorrect.", "error")
            elif new_password != confirm_password:
                flash("New passwords do not match.", "error")
            else:
                user['password'] = new_password  # In real app, hash the password
                flash("Password updated successfully!", "success")

        # -------- Notification Preferences --------
        user['notifications'] = {
            "email": bool(request.form.get('email_notifications')),
            "sms": bool(request.form.get('sms_notifications')),
            "gd_reminders": bool(request.form.get('gd_reminders'))
        }
        flash("Settings updated successfully!", "success")

        # Save back to session
        session['user'] = user

    return render_template('settings.html', username=user.get('name', 'User'))
