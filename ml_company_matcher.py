"""
ML-Based Company Recommendation System
Uses machine learning to match students with companies
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
from typing import List, Dict, Any

DB_PATH = "users.db"

class CompanyRecommendationEngine:
    """Machine Learning engine for company recommendations"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'cgpa', 'graduation_year', 'python_skill', 'java_skill', 'cpp_skill',
            'javascript_skill', 'sql_skill', 'machine_learning_skill', 'data_structures_skill',
            'algorithms_skill', 'web_development_skill', 'mobile_development_skill',
            'cloud_computing_skill', 'ai_skill', 'data_science_skill'
        ]
        self.model_path = "models/company_recommendation_model.pkl"
        self.encoder_path = "models/company_encoder.pkl"
    
    def extract_skills_features(self, skills_string: str) -> Dict[str, int]:
        """Extract binary features for each skill"""
        if not skills_string:
            return {col: 0 for col in self.feature_columns[2:]}
        
        skills = [skill.strip().lower() for skill in skills_string.split(',')]
        
        skill_mapping = {
            'python_skill': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'java_skill': ['java', 'spring', 'hibernate', 'j2ee'],
            'cpp_skill': ['c++', 'cpp', 'c'],
            'javascript_skill': ['javascript', 'node.js', 'react', 'angular', 'vue'],
            'sql_skill': ['sql', 'mysql', 'postgresql', 'oracle'],
            'machine_learning_skill': ['machine learning', 'ml', 'tensorflow', 'pytorch', 'scikit-learn'],
            'data_structures_skill': ['data structures', 'algorithms', 'ds'],
            'algorithms_skill': ['algorithms', 'algo', 'sorting', 'searching'],
            'web_development_skill': ['html', 'css', 'bootstrap', 'web development'],
            'mobile_development_skill': ['android', 'ios', 'flutter', 'react native', 'kotlin', 'swift'],
            'cloud_computing_skill': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes'],
            'ai_skill': ['artificial intelligence', 'ai', 'deep learning', 'neural networks'],
            'data_science_skill': ['data science', 'data analysis', 'statistics', 'analytics']
        }
        
        features = {}
        for feature_name, keywords in skill_mapping.items():
            features[feature_name] = 1 if any(keyword in skills for keyword in keywords) else 0
        
        return features
    
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data from historical placements"""
        conn = sqlite3.connect(DB_PATH)
        
        # Create synthetic training data based on company requirements
        training_data = []
        
        # TCS - prefers Java, SQL, basic programming
        for _ in range(100):
            training_data.append({
                'cgpa': np.random.normal(6.5, 0.5),
                'graduation_year': 2024,
                'python_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'java_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'cpp_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'javascript_skill': np.random.choice([0, 1], p=[0.5, 0.5]),
                'sql_skill': np.random.choice([0, 1], p=[0.1, 0.9]),
                'machine_learning_skill': np.random.choice([0, 1], p=[0.7, 0.3]),
                'data_structures_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'algorithms_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'web_development_skill': np.random.choice([0, 1], p=[0.6, 0.4]),
                'mobile_development_skill': np.random.choice([0, 1], p=[0.8, 0.2]),
                'cloud_computing_skill': np.random.choice([0, 1], p=[0.6, 0.4]),
                'ai_skill': np.random.choice([0, 1], p=[0.8, 0.2]),
                'data_science_skill': np.random.choice([0, 1], p=[0.7, 0.3]),
                'company_id': 1  # TCS
            })
        
        # Infosys - prefers Python, ML, data science
        for _ in range(100):
            training_data.append({
                'cgpa': np.random.normal(6.8, 0.4),
                'graduation_year': 2024,
                'python_skill': np.random.choice([0, 1], p=[0.1, 0.9]),
                'java_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'cpp_skill': np.random.choice([0, 1], p=[0.6, 0.4]),
                'javascript_skill': np.random.choice([0, 1], p=[0.5, 0.5]),
                'sql_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'machine_learning_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'data_structures_skill': np.random.choice([0, 1], p=[0.1, 0.9]),
                'algorithms_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'web_development_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'mobile_development_skill': np.random.choice([0, 1], p=[0.7, 0.3]),
                'cloud_computing_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'ai_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'data_science_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'company_id': 2  # Infosys
            })
        
        # Wipro - balanced skills
        for _ in range(100):
            training_data.append({
                'cgpa': np.random.normal(6.6, 0.5),
                'graduation_year': 2024,
                'python_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'java_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'cpp_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'javascript_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'sql_skill': np.random.choice([0, 1], p=[0.1, 0.9]),
                'machine_learning_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'data_structures_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'algorithms_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'web_development_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'mobile_development_skill': np.random.choice([0, 1], p=[0.5, 0.5]),
                'cloud_computing_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'ai_skill': np.random.choice([0, 1], p=[0.5, 0.5]),
                'data_science_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'company_id': 3  # Wipro
            })
        
        # Tech Mahindra - prefers testing, quality assurance
        for _ in range(100):
            training_data.append({
                'cgpa': np.random.normal(6.4, 0.6),
                'graduation_year': 2024,
                'python_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'java_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'cpp_skill': np.random.choice([0, 1], p=[0.5, 0.5]),
                'javascript_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'sql_skill': np.random.choice([0, 1], p=[0.1, 0.9]),
                'machine_learning_skill': np.random.choice([0, 1], p=[0.6, 0.4]),
                'data_structures_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'algorithms_skill': np.random.choice([0, 1], p=[0.5, 0.5]),
                'web_development_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'mobile_development_skill': np.random.choice([0, 1], p=[0.6, 0.4]),
                'cloud_computing_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'ai_skill': np.random.choice([0, 1], p=[0.7, 0.3]),
                'data_science_skill': np.random.choice([0, 1], p=[0.5, 0.5]),
                'company_id': 4  # Tech Mahindra
            })
        
        # Accenture - consulting, business analysis
        for _ in range(100):
            training_data.append({
                'cgpa': np.random.normal(7.0, 0.4),
                'graduation_year': 2024,
                'python_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'java_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'cpp_skill': np.random.choice([0, 1], p=[0.6, 0.4]),
                'javascript_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'sql_skill': np.random.choice([0, 1], p=[0.1, 0.9]),
                'machine_learning_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'data_structures_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'algorithms_skill': np.random.choice([0, 1], p=[0.3, 0.7]),
                'web_development_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'mobile_development_skill': np.random.choice([0, 1], p=[0.7, 0.3]),
                'cloud_computing_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'ai_skill': np.random.choice([0, 1], p=[0.4, 0.6]),
                'data_science_skill': np.random.choice([0, 1], p=[0.2, 0.8]),
                'company_id': 5  # Accenture
            })
        
        df = pd.DataFrame(training_data)
        conn.close()
        return df
    
    def train_model(self):
        """Train the recommendation model"""
        print("🤖 Training company recommendation model...")
        
        # Prepare training data
        df = self.prepare_training_data()
        
        # Split features and target
        X = df[self.feature_columns]
        y = df['company_id']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model accuracy: {accuracy:.2f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.label_encoder, self.encoder_path)
        
        return accuracy
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            self.model = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.encoder_path)
            return True
        return False
    
    def predict_company_match(self, student_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict company matches for a student"""
        if not self.model:
            if not self.load_model():
                self.train_model()
        
        # Extract features
        features = {
            'cgpa': student_data.get('cgpa', 7.0),
            'graduation_year': student_data.get('graduation_year', 2024)
        }
        
        # Add skill features
        skills_features = self.extract_skills_features(student_data.get('skills', ''))
        features.update(skills_features)
        
        # Create feature vector
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Get predictions with probabilities
        probabilities = self.model.predict_proba(feature_vector)[0]
        company_ids = self.model.classes_
        
        # Get company details
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM companies")
        companies = c.fetchall()
        conn.close()
        
        # Create company mapping
        company_map = {comp[0]: comp for comp in companies}
        
        # Generate recommendations
        recommendations = []
        for i, company_id in enumerate(company_ids):
            if company_id in company_map:
                comp = company_map[company_id]
                recommendations.append({
                    "id": comp[0],
                    "name": comp[1],
                    "role": comp[5],
                    "package": comp[6],
                    "location": comp[7],
                    "industry": comp[8],
                    "min_cgpa": comp[2],
                    "required_skills": comp[3],
                    "match_probability": round(probabilities[i] * 100, 2),
                    "recommendation_type": "ml_based"
                })
        
        # Sort by match probability
        recommendations.sort(key=lambda x: x['match_probability'], reverse=True)
        
        return recommendations[:10]  # Return top 10 matches

def ml_recommend_companies(student_email: str, model_type: str = "random_forest") -> List[Dict[str, Any]]:
    """Main function to get ML-based company recommendations"""
    try:
        # Get student data
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM students WHERE email = ?", (student_email,))
        student = c.fetchone()
        conn.close()
        
        if not student:
            return []
        
        # Prepare student data
        student_data = {
            "name": student[2],
            "email": student[1],
            "cgpa": student[3],
            "graduation_year": student[4],
            "skills": student[5]
        }
        
        # Get ML recommendations
        engine = CompanyRecommendationEngine()
        recommendations = engine.predict_company_match(student_data)
        
        return recommendations
        
    except Exception as e:
        print(f"Error in ML recommendation: {e}")
        return []

if __name__ == "__main__":
    print("🚀 Training ML Company Recommendation Model")
    print("=" * 50)
    
    engine = CompanyRecommendationEngine()
    accuracy = engine.train_model()
    
    print(f"✅ Model trained with {accuracy:.2f} accuracy")
    print("📁 Model saved to models/ directory")
    print("=" * 50)