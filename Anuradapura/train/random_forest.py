import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

print("--- FINAL ML PIPELINE: RANDOM FOREST ---")

# 1. LOAD DATASET
# Ensure this matches your final cleaned CSV file
df = pd.read_csv('anuradhapura_final_ml_dataset.csv') 

# 2. DEFINE FEATURES
features_to_keep = [
    'Road_Type', 'Sinuosity_Index', 'Dist_to_Intersection_m', 'No_of_Lanes', 
    'Speed_Limit_kmh', 'Has_Zebra_Crossing', 'Has_Streetlight_Infrastructure', 
    'Baseline_Traffic_Volume', 'Elevation_Gradient_Pct', 'Topography', 'Road_Condition'
]

# 3. ENCODE CATEGORICAL VARIABLES
# This turns text (e.g., 'primary', 'Flat') into numbers (0, 1, 2)
# This is required because Random Forest cannot read raw text.
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
# 'balanced' class_weight tells the model to pay extra attention to the minority class (Hotspots)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# 6. PREDICT & EVALUATE
y_pred = rf.predict(X_test)

print("\n=== ACADEMIC PERFORMANCE METRICS ===")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score:          {f1_score(y_test, y_pred) * 100:.2f}%")
print(f"Standard Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred, target_names=["Safe Road", "Hotspot"]))

print("\n5. Saving Testing Dataset as CSV...")

# Combine X_test and y_test back together
# We use X_test.index to ensure we pull the correct rows from your original df
test_df = df.loc[X_test.index].copy()

# Add the actual result (y_test) and your model's prediction (y_pred)
test_df['Actual_Hotspot'] = y_test
test_df['Predicted_Hotspot'] = y_pred

# Save to CSV
output_filename = 'anuradhapura_testing_results_basic_RF.csv'
test_df.to_csv(output_filename, index=False)

print(f"✅ Success! Testing dataset with predictions saved as '{output_filename}'")

# 7. CONFUSION MATRIX VISUALIZATION
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('Confusion_Matrix_basic_RF.png', dpi=300)

print("\n✅ Pipeline complete. Confusion matrix saved as 'Final_Confusion_Matrix.png'.")