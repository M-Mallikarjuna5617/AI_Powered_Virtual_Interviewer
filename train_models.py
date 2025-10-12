"""
Train Decision Tree and Random Forest models for company recommendation.
Run this script once to generate the models.
"""

from ml_company_matcher import train_models, get_feature_importance

print("=" * 60)
print("🚀 Training ML Models for Company Recommendation")
print("=" * 60)

# Train models
dt_model, rf_model, le = train_models()

print("\n" + "=" * 60)
print("📊 Feature Importance Analysis")
print("=" * 60)

importance = get_feature_importance()
for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 50)
    print(f"{feature:15} {bar} {imp:.3f}")

print("\n" + "=" * 60)
print("✅ Model training complete!")
print("=" * 60)
print("\nModels saved in 'models/' directory:")
print("  - dt_model.pkl (Decision Tree)")
print("  - rf_model.pkl (Random Forest)")
print("  - label_encoder.pkl (Label Encoder)")
print("\n🎯 You can now use ML-based company recommendations!")