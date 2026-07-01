import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

print("--- FINAL ML PIPELINE: XGBOOST ---")

# 1. LOAD DATASET
df = pd.read_csv('anuradhapura_final_ml_dataset.csv') 

# 2. DEFINE FEATURES
features_to_keep = [
    'Road_Type', 'Sinuosity_Index', 'Dist_to_Intersection_m', 'No_of_Lanes', 
    'Speed_Limit_kmh', 'Has_Zebra_Crossing', 'Has_Streetlight_Infrastructure', 
    'Baseline_Traffic_Volume', 'Elevation_Gradient_Pct', 'Topography', 'Road_Condition'
]

# 3. ENCODE CATEGORICAL VARIABLES
encoders = {}
categorical_cols = ['Road_Type', 'Road_Surface', 'Baseline_Traffic_Volume', 'Topography', 'Road_Condition']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# 4. PREPARE DATA
X = df[features_to_keep]
y = df['Is_Hotspot']

print(f"Features in use: {features_to_keep}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. TRAIN MODEL
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

# 6. PREDICT & EVALUATE
y_pred = xgb.predict(X_test)

print("\n=== ACADEMIC PERFORMANCE METRICS ===")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Standard Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision:         {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall:            {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score:          {f1_score(y_test, y_pred) * 100:.2f}%")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred, target_names=["Safe Road", "Hotspot"]))

print("\n5. Saving Testing Dataset as CSV...")

# Combine X_test and y_test back together
test_df = df.loc[X_test.index].copy()

# Add the actual result (y_test) and your model's prediction (y_pred)
test_df['Actual_Hotspot'] = y_test
test_df['Predicted_Hotspot'] = y_pred

# Save to CSV
output_filename = 'anuradhapura_testing_results_basic_XGB.csv'
test_df.to_csv(output_filename, index=False)

print(f"Success! Testing dataset with predictions saved as '{output_filename}'")

# 7. CONFUSION MATRIX VISUALIZATION
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('XGBoost - Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('Confusion_Matrix_basic_XGB.png', dpi=300)

print("\nPipeline complete. Confusion matrix saved as 'Confusion_Matrix_basic_XGB.png'.")

# 8. FEATURE IMPORTANCE BAR CHART
print("\n8. Saving Feature Importance Chart...")
feature_names = X.columns
importances = xgb.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#1565C0')
plt.xlabel('Importance')
plt.title('XGBoost Basic - Feature Importance')
plt.tight_layout()
plt.savefig('Feature_Importance_basic_XGB.png', dpi=300)
print("Saved -> 'Feature_Importance_basic_XGB.png'")
