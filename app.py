from flask import Flask
import os

from auth import auth_bp, init_oauth
from routes import routes_bp
from dashboard import dashboard_bp
from database import init_db

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ALWAYS Init DB (it uses CREATE TABLE IF NOT EXISTS, so it's safe)
init_db()

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(routes_bp)
app.register_blueprint(dashboard_bp, url_prefix='/dashboard')

# Init OAuth
init_oauth(app)

if __name__ == "__main__":
    app.run(debug=True)