import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 

print("--- SCRIPT 3: SMOTE + TUNED RANDOM FOREST ---")
print("1. Loading dataset...")
df = pd.read_csv('anuradhapura_final_ml_dataset.csv')

features_to_keep = [
    'Road_Type', 'Sinuosity_Index', 'Dist_to_Intersection_m', 'No_of_Lanes', 
    'Speed_Limit_kmh', 'Has_Zebra_Crossing', 'Has_Streetlight_Infrastructure', 
    'Baseline_Traffic_Volume', 'Elevation_Gradient_Pct', 'Topography'
]

X = pd.get_dummies(df[features_to_keep], drop_first=True)
y = df['Is_Hotspot']

print("2. Splitting Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Tuning SMOTE + Random Forest with GridSearch...")
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

# Fixed: Using higher SMOTE ratios for Ella to avoid the FitFailedWarning!
param_grid = {
    'smote__sampling_strategy': [0.3, 0.5, 0.7, 'auto'], 
    'rf__n_estimators': [100, 200],         
    'rf__max_depth': [3, 5],
    'rf__min_samples_leaf': [5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

print("\n4. Generating Predictions...")
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

print("\n=== METRICS ===")
print(f"Training Balanced Accuracy: {balanced_accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Balanced Accuracy:  {balanced_accuracy_score(y_test, test_pred) * 100:.2f}%")
print(f"Training F1 Score:          {f1_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing F1 Score:           {f1_score(y_test, test_pred) * 100:.2f}%")
print(f"Training Accuracy (Std):    {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Accuracy (Std):     {accuracy_score(y_test, test_pred) * 100:.2f}%")

print("\n5. Saving QGIS CSV (Unseen Test Data)...")
test_df = df.loc[X_test.index].copy()
test_df['Predicted_Hotspot'] = test_pred

def categorize_prediction(row):
    if row['Is_Hotspot'] == 1 and row['Predicted_Hotspot'] == 1: return 'True Hotspot'
    if row['Is_Hotspot'] == 0 and row['Predicted_Hotspot'] == 0: return 'True Safe'
    if row['Is_Hotspot'] == 0 and row['Predicted_Hotspot'] == 1: return 'False Alarm'
    if row['Is_Hotspot'] == 1 and row['Predicted_Hotspot'] == 0: return 'Missed Hotspot'

test_df['Prediction_Category'] = test_df.apply(categorize_prediction, axis=1)
test_df.to_csv('QGIS_Predictions_3_SMOTE_Tuned.csv', index=False)
print("Saved -> 'QGIS_Predictions_3_SMOTE_Tuned.csv'")

print("\n6. Saving Real-World Confusion Matrix Image...")
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('Real-World Test Data - Confusion Matrix')
plt.ylabel('True Reality (Map)')
plt.xlabel('Algorithm Prediction')
plt.tight_layout()
plt.savefig('Matrix_3_SMOTE_Tuned_RF.png', dpi=300)
print("Saved -> 'Matrix_3_SMOTE_Tuned_RF.png'")

# ==========================================
# --- NEW: SMOTE SYNTHETIC DATA EXPORT ---
# ==========================================
print("\n7. Extracting and Saving SMOTE Synthetic Data...")
best_smote_ratio = grid_search.best_params_['smote__sampling_strategy']

# Re-run SMOTE manually to get the synthetic rows
manual_smote = SMOTE(sampling_strategy=best_smote_ratio, random_state=42)
X_train_resampled, y_train_resampled = manual_smote.fit_resample(X_train, y_train)

# Make predictions on this new mixed dataset (Real Train + Fake SMOTE)
resampled_predictions = best_model.predict(X_train_resampled)

smote_export_df = X_train_resampled.copy()
smote_export_df['Is_Hotspot_Actual'] = y_train_resampled
smote_export_df['Predicted_Hotspot'] = resampled_predictions

# Tag which rows are real and which are synthetic
num_real_rows = len(X_train)
num_synthetic_rows = len(X_train_resampled) - num_real_rows
smote_export_df['Data_Type'] = ['Real'] * num_real_rows + ['Synthetic'] * num_synthetic_rows

smote_export_df.to_csv('SMOTE_Combined_Training_Data.csv', index=False)
print(f"Saved -> 'SMOTE_Combined_Training_Data.csv'")
print(f"         ({num_real_rows} Real rows and {num_synthetic_rows} Synthetic rows)")

print("\n8. Saving SMOTE Training Confusion Matrix Image...")
cm_smote = confusion_matrix(y_train_resampled, resampled_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', 
            xticklabels=["Predicted Safe", "Predicted Hotspot"], 
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('Training Data (with SMOTE) - Confusion Matrix')
plt.ylabel('SMOTE Reality (Real + Fake)')
plt.xlabel('Algorithm Prediction')
plt.tight_layout()
plt.savefig('Matrix_4_SMOTE_Training_Data.png', dpi=300)
print("Saved -> 'Matrix_4_SMOTE_Training_Data.png'")
print("\nSuccess! All files generated perfectly.")