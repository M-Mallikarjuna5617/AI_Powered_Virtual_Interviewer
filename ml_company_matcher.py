import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# -------------------- Generate Training Data --------------------
def generate_training_data():
    """
    Generate synthetic training data based on historical placement patterns.
    In production, replace this with real historical data.
    """
    data = []
    
    # Define company profiles and their typical candidate profiles
    company_profiles = {
        "Google": {"min_cgpa": 8.0, "preferred_skills": ["Python", "Machine Learning", "Algorithms"], "difficulty": "high"},
        "Microsoft": {"min_cgpa": 7.5, "preferred_skills": ["C++", "Cloud", "Data Structures"], "difficulty": "high"},
        "Amazon": {"min_cgpa": 7.5, "preferred_skills": ["Java", "AWS", "System Design"], "difficulty": "high"},
        "TCS": {"min_cgpa": 6.0, "preferred_skills": ["Java", "SQL"], "difficulty": "low"},
        "Infosys": {"min_cgpa": 6.5, "preferred_skills": ["Python", "Java"], "difficulty": "low"},
        "Flipkart": {"min_cgpa": 7.0, "preferred_skills": ["Java", "System Design"], "difficulty": "medium"},
        "Razorpay": {"min_cgpa": 7.2, "preferred_skills": ["Python", "API"], "difficulty": "medium"},
    }
    
    # Generate 500+ training samples
    for _ in range(500):
        cgpa = round(np.random.uniform(5.5, 9.5), 2)
        num_skills = np.random.randint(2, 8)
        has_ml = np.random.choice([0, 1], p=[0.6, 0.4])
        has_dsa = np.random.choice([0, 1], p=[0.4, 0.6])
        has_cloud = np.random.choice([0, 1], p=[0.5, 0.5])
        
        for company, profile in company_profiles.items():
            # Calculate match probability based on CGPA and skills
            cgpa_match = 1 if cgpa >= profile["min_cgpa"] else 0
            skill_bonus = num_skills / 10
            
            # Calculate selection probability
            if profile["difficulty"] == "high":
                base_prob = 0.2 if cgpa >= 8.0 else 0.05
            elif profile["difficulty"] == "medium":
                base_prob = 0.4 if cgpa >= 7.0 else 0.15
            else:
                base_prob = 0.7 if cgpa >= 6.0 else 0.3
            
            selection_prob = min(base_prob + skill_bonus + (has_dsa * 0.15) + (has_ml * 0.1), 0.95)
            selected = np.random.choice([0, 1], p=[1-selection_prob, selection_prob])
            
            data.append({
                "company": company,
                "cgpa": cgpa,
                "num_skills": num_skills,
                "has_ml": has_ml,
                "has_dsa": has_dsa,
                "has_cloud": has_cloud,
                "selected": selected
            })
    
    return pd.DataFrame(data)


# -------------------- Train Models --------------------
def train_models():
    """Train Decision Tree and Random Forest models."""
    print("Generating training data...")
    df = generate_training_data()
    
    # Encode company names
    le = LabelEncoder()
    df['company_encoded'] = le.fit_transform(df['company'])
    
    # Features and target
    X = df[['company_encoded', 'cgpa', 'num_skills', 'has_ml', 'has_dsa', 'has_cloud']]
    y = df['selected']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_accuracy = dt_model.score(X_test, y_test)
    print(f"Decision Tree Accuracy: {dt_accuracy:.2%}")
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    with open('models/dt_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("✅ Models saved successfully!")
    return dt_model, rf_model, le


# -------------------- Load Models --------------------
def load_models():
    """Load trained models."""
    try:
        with open('models/dt_model.pkl', 'rb') as f:
            dt_model = pickle.load(f)
        with open('models/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return dt_model, rf_model, le
    except FileNotFoundError:
        print("Models not found. Training new models...")
        return train_models()


# -------------------- Extract Features from Student --------------------
def extract_student_features(student_skills, student_cgpa):
    """Extract ML features from student profile."""
    skills_lower = [s.strip().lower() for s in student_skills.split(",")]
    
    features = {
        "cgpa": student_cgpa,
        "num_skills": len(skills_lower),
        "has_ml": int(any(skill in skills_lower for skill in ["machine learning", "ml", "deep learning", "ai"])),
        "has_dsa": int(any(skill in skills_lower for skill in ["data structures", "algorithms", "dsa"])),
        "has_cloud": int(any(skill in skills_lower for skill in ["aws", "azure", "cloud", "docker", "kubernetes"]))
    }
    
    return features


# -------------------- ML-Based Company Recommendation --------------------
def ml_recommend_companies(student_email, model_type="random_forest"):
    """
    Recommend companies using ML models.
    
    Args:
        student_email: Student's email
        model_type: "decision_tree" or "random_forest"
    
    Returns:
        List of recommended companies with selection probability
    """
    import sqlite3
    
    # Load models
    dt_model, rf_model, le = load_models()
    model = rf_model if model_type == "random_forest" else dt_model
    
    # Get student data
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE email=?", (student_email,))
    student = c.fetchone()
    
    if not student:
        conn.close()
        return []
    
    _, email, name, cgpa, grad_year, skills = student
    
    # Extract features
    student_features = extract_student_features(skills, cgpa)
    
    # Get all companies
    c.execute("SELECT * FROM companies WHERE graduation_year=?", (grad_year,))
    companies = c.fetchall()
    conn.close()
    
    recommendations = []
    
    for comp in companies:
        comp_id, cname, min_cgpa, req_skills, comp_year, role, package, location, industry = comp
        
        # Check if company exists in training data
        try:
            company_encoded = le.transform([cname])[0]
        except:
            # If company not in training data, use basic filtering
            if cgpa >= min_cgpa:
                student_skills_set = set([s.strip().lower() for s in skills.split(",")])
                required_skills_set = set([s.strip().lower() for s in (req_skills or "").split(",")])
                matched_skills = list(student_skills_set.intersection(required_skills_set))
                
                recommendations.append({
                    "id": comp_id,
                    "name": cname,
                    "role": role,
                    "package": package,
                    "location": location,
                    "industry": industry,
                    "min_cgpa": min_cgpa,
                    "required_skills": req_skills,
                    "matched_skills": matched_skills,
                    "selection_probability": 50.0,  # Default probability
                    "recommendation_type": "rule_based"
                })
            continue
        
        # Prepare features for prediction
        X = [[
            company_encoded,
            student_features["cgpa"],
            student_features["num_skills"],
            student_features["has_ml"],
            student_features["has_dsa"],
            student_features["has_cloud"]
        ]]
        
        # Predict selection probability
        if model_type == "random_forest":
            prob = model.predict_proba(X)[0][1]  # Probability of class 1 (selected)
        else:
            prob = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else model.predict(X)[0]
        
        # Only recommend if probability > 30%
        if prob >= 0.3:
            # Get matched skills for display
            student_skills_set = set([s.strip().lower() for s in skills.split(",")])
            required_skills_set = set([s.strip().lower() for s in (req_skills or "").split(",")])
            matched_skills = list(student_skills_set.intersection(required_skills_set))
            
            recommendations.append({
                "id": comp_id,
                "name": cname,
                "role": role,
                "package": package,
                "location": location,
                "industry": industry,
                "min_cgpa": min_cgpa,
                "required_skills": req_skills,
                "matched_skills": matched_skills,
                "selection_probability": round(prob * 100, 1),
                "recommendation_type": model_type,
                "confidence": "high" if prob >= 0.7 else "medium" if prob >= 0.5 else "low"
            })
    
    # Sort by selection probability
    recommendations.sort(key=lambda x: x['selection_probability'], reverse=True)
    
    return recommendations


# -------------------- Get Feature Importance --------------------
def get_feature_importance():
    """Get feature importance from Random Forest model."""
    dt_model, rf_model, le = load_models()
    
    feature_names = ['company', 'cgpa', 'num_skills', 'has_ml', 'has_dsa', 'has_cloud']
    importances = rf_model.feature_importances_
    
    importance_dict = {}
    for name, importance in zip(feature_names, importances):
        importance_dict[name] = round(importance, 3)
    
    return importance_dict


if __name__ == "__main__":
    # Train models
    train_models()
    
    # Show feature importance
    print("\n📊 Feature Importance:")
    for feature, importance in get_feature_importance().items():
        print(f"  {feature}: {importance}")