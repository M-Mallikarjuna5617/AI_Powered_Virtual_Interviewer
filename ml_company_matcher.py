"""
Advanced Company Recommendation Algorithm
Uses multiple factors: CGPA, Skills, Backlogs, Resume Analysis
"""

import sqlite3
import re
from typing import List, Dict, Any

DB_PATH = "users.db"

class CompanyRecommendationEngine:
    
    def __init__(self, student_email: str):
        self.student_email = student_email
        self.conn = sqlite3.connect(DB_PATH)
        self.c = self.conn.cursor()
        
    def get_student_profile(self) -> Dict[str, Any]:
        """Get complete student profile"""
        self.c.execute("SELECT * FROM students WHERE email=?", (self.student_email,))
        student = self.c.fetchone()
        
        if not student:
            return None
        
        return {
            "email": student[1],
            "name": student[2],
            "cgpa": student[3] or 6.0,
            "graduation_year": student[4] or 2025,
            "skills": student[5] or "",
            "active_backlogs": 0  # Can be extended to track from DB
        }
    
    def calculate_skill_match(self, student_skills: str, required_skills: str) -> Dict[str, Any]:
        """Calculate detailed skill matching score"""
        student_skill_set = set([s.strip().lower() for s in student_skills.split(",") if s.strip()])
        required_skill_set = set([s.strip().lower() for s in required_skills.split(",") if s.strip()])
        
        if not required_skill_set:
            return {"score": 50.0, "matched": [], "missing": []}
        
        matched_skills = student_skill_set.intersection(required_skill_set)
        missing_skills = required_skill_set - student_skill_set
        
        match_percentage = (len(matched_skills) / len(required_skill_set)) * 100
        
        return {
            "score": round(match_percentage, 1),
            "matched": list(matched_skills),
            "missing": list(missing_skills),
            "total_required": len(required_skill_set),
            "total_matched": len(matched_skills)
        }
    
    def calculate_eligibility_score(self, student: Dict, company: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive eligibility score
        Factors: CGPA (40%), Skills (40%), Backlogs (20%)
        """
        # CGPA Score (40%)
        cgpa_score = 0
        if student["cgpa"] >= company["min_cgpa"]:
            cgpa_diff = student["cgpa"] - company["min_cgpa"]
            cgpa_score = min(40, 25 + (cgpa_diff * 5))  # Base 25, bonus for higher CGPA
        
        # Skills Score (40%)
        skill_match = self.calculate_skill_match(student["skills"], company["required_skills"])
        skills_score = (skill_match["score"] / 100) * 40
        
        # Backlog Score (20%)
        backlog_score = 0
        if student.get("active_backlogs", 0) <= company.get("active_backlogs", 0):
            backlog_score = 20
        elif student.get("active_backlogs", 0) <= company.get("active_backlogs", 0) + 1:
            backlog_score = 10
        
        total_score = cgpa_score + skills_score + backlog_score
        
        return {
            "total_score": round(total_score, 1),
            "cgpa_score": round(cgpa_score, 1),
            "skills_score": round(skills_score, 1),
            "backlog_score": round(backlog_score, 1),
            "skill_match": skill_match,
            "is_eligible": total_score >= 50 and student["cgpa"] >= company["min_cgpa"]
        }
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get ranked company recommendations"""
        student = self.get_student_profile()
        
        if not student:
            return []
        
        # Get all companies for student's graduation year
        self.c.execute("""
            SELECT * FROM companies 
            WHERE graduation_year = ?
            ORDER BY min_cgpa ASC
        """, (student["graduation_year"],))
        
        companies = self.c.fetchall()
        recommendations = []
        
        for comp in companies:
            company_data = {
                "id": comp[0],
                "name": comp[1],
                "min_cgpa": comp[2],
                "required_skills": comp[3] or "",
                "graduation_year": comp[4],
                "role": comp[5],
                "package_offered": comp[6],
                "location": comp[7],
                "industry": comp[8],
                "eligibility_criteria": comp[9],
                "selection_process": comp[10],
                "no_of_rounds": comp[11],
                "active_backlogs": comp[12] or 0,
                "company_description": comp[13],
                "last_visited_year": comp[14]
            }
            
            # Calculate eligibility
            eligibility = self.calculate_eligibility_score(student, company_data)
            
            if eligibility["is_eligible"]:
                # Determine recommendation strength
                score = eligibility["total_score"]
                if score >= 80:
                    strength = "Excellent Match"
                    strength_class = "success"
                elif score >= 65:
                    strength = "Good Match"
                    strength_class = "primary"
                elif score >= 50:
                    strength = "Fair Match"
                    strength_class = "warning"
                else:
                    continue  # Skip companies below 50% match
                
                recommendations.append({
                    **company_data,
                    "eligibility_score": eligibility["total_score"],
                    "cgpa_score": eligibility["cgpa_score"],
                    "skills_score": eligibility["skills_score"],
                    "backlog_score": eligibility["backlog_score"],
                    "skill_match_percentage": eligibility["skill_match"]["score"],
                    "matched_skills": eligibility["skill_match"]["matched"],
                    "missing_skills": eligibility["skill_match"]["missing"],
                    "recommendation_strength": strength,
                    "strength_class": strength_class
                })
        
        # Sort by eligibility score
        recommendations.sort(key=lambda x: x["eligibility_score"], reverse=True)
        
        self.conn.close()
        return recommendations
    
    def get_company_statistics(self, company_id: int) -> Dict[str, Any]:
        """Get statistics for a specific company"""
        self.c.execute("""
            SELECT 
                COUNT(*) as total_attempts,
                AVG(score) as avg_score,
                test_type
            FROM test_history
            WHERE company_id = ?
            GROUP BY test_type
        """, (company_id,))
        
        stats = {}
        for row in self.c.fetchall():
            stats[row[2]] = {
                "total_attempts": row[0],
                "average_score": round(row[1], 2) if row[1] else 0
            }
        
        return stats


def recommend_companies_for_student(student_email: str) -> List[Dict[str, Any]]:
    """Main function to get company recommendations"""
    engine = CompanyRecommendationEngine(student_email)
    return engine.get_recommendations()


# Alternative: ML-Based Recommendation (if models are trained)
def ml_recommend_companies(student_email: str):
    """ML-based recommendation using trained models"""
    try:
        from ml_company_matcher import ml_recommend_companies as ml_rec
        return ml_rec(student_email, model_type="random_forest")
    except Exception as e:
        print(f"ML recommendation failed: {e}")
        return recommend_companies_for_student(student_email)


def get_hybrid_recommendations(student_email: str) -> List[Dict[str, Any]]:
    """
    Hybrid approach: Combine rule-based and ML recommendations
    """
    # Get rule-based recommendations
    rule_based = recommend_companies_for_student(student_email)
    
    # Try ML recommendations
    try:
        from ml_company_matcher import ml_recommend_companies as ml_rec
        ml_based = ml_rec(student_email, model_type="random_forest")
        
        # Merge results - prioritize ML scores but keep rule-based details
        company_map = {comp["id"]: comp for comp in rule_based}
        
        for ml_comp in ml_based:
            comp_id = ml_comp["id"]
            if comp_id in company_map:
                # Combine scores: 60% ML, 40% rule-based
                ml_score = ml_comp.get("selection_probability", 50)
                rule_score = company_map[comp_id].get("eligibility_score", 50)
                combined_score = (ml_score * 0.6) + (rule_score * 0.4)
                
                company_map[comp_id]["final_score"] = round(combined_score, 1)
                company_map[comp_id]["ml_probability"] = ml_score
                company_map[comp_id]["recommendation_type"] = "hybrid"
        
        # Sort by final score
        results = sorted(company_map.values(), key=lambda x: x.get("final_score", x.get("eligibility_score", 0)), reverse=True)
        return results
        
    except:
        # Fallback to rule-based only
        return rule_based


if __name__ == "__main__":
    # Test the recommendation engine
    test_email = "test@student.com"
    recommendations = recommend_companies_for_student(test_email)
    
    print(f"\n📊 Company Recommendations for {test_email}")
    print("=" * 80)
    
    for idx, comp in enumerate(recommendations, 1):
        print(f"\n{idx}. {comp['name']} - {comp['role']}")
        print(f"   Package: {comp['package_offered']}")
        print(f"   Location: {comp['location']}")
        print(f"   Eligibility Score: {comp['eligibility_score']}/100")
        print(f"   Skill Match: {comp['skill_match_percentage']}%")
        print(f"   Recommendation: {comp['recommendation_strength']}")
        print(f"   Matched Skills: {', '.join(comp['matched_skills'][:5])}")