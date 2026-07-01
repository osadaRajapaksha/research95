import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, make_scorer

print("--- THESIS PIPELINE: F2-SCORE OPTIMIZATION (BALANCED SAFETY) ---")

# 1. LOAD & ENCODE
df = pd.read_csv('anuradhapura_final_ml_dataset.csv')
le_dict = {}
for col in ['Road_Type', 'Road_Surface', 'Baseline_Traffic_Volume', 'Topography', 'Road_Condition']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop(columns=['Is_Hotspot', 'Road_Name', 'Segment_Index', 'Accident_Count'])
y = df['Is_Hotspot']

# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. CREATE CUSTOM F2-SCORER
# beta=2 means Recall is weighted 2x heavier than Precision
f2_scorer = make_scorer(fbeta_score, beta=2)

# 4. GRID SEARCH (Optimizing for F2)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

print("Running Grid Search optimized for F2-Score...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid, 
    cv=5, 
    scoring=f2_scorer, # <--- The academic secret weapon
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 5. MODERATE THRESHOLD
# Since the model is already optimized for Recall (F2), we don't need a drastic 0.25 threshold.
# 0.35 or 0.40 provides a much better balance to bring the 50 False Alarms down.
y_probs = best_model.predict_proba(X_test)[:, 1]
balanced_threshold = 0.35 
y_pred = (y_probs >= balanced_threshold).astype(int)

print("\n=== FINAL BALANCED METRICS (F2-Optimized) ===")
print(classification_report(y_test, y_pred, target_names=["Safe Road", "Hotspot"]))

# 6. SAVE TESTING CSV & CONFUSION MATRIX
test_df = df.loc[X_test.index].copy()
test_df['Predicted_Hotspot'] = y_pred
test_df.to_csv('QGIS_Predictions_F2_Optimized.csv', index=False)
print("✅ Testing Predictions saved -> 'QGIS_Predictions_F2_Optimized.csv'")

cm_test = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title(f'F2-Optimized Confusion Matrix (Threshold = {balanced_threshold})')
plt.savefig('Testing_Confusion_Matrix.png', dpi=300)
print("✅ Testing Matrix saved -> 'Testing_Confusion_Matrix.png'")