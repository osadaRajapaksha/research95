import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

print("--- SCRIPT: BAYESIAN OPTIMIZED SMOTE + TUNED XGBOOST ---")
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Tuning SMOTE + XGBoost with Bayesian Optimization...")
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),  # 'auto' = fully balance classes
    ('xgb', XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    ))
])

# Bayesian search space — only model hyperparameters
search_space = {
    'xgb__n_estimators':        Integer(50, 200),
    'xgb__max_depth':           Integer(2, 5),
    'xgb__learning_rate':       Real(0.01, 0.1, prior='log-uniform'),
    'xgb__subsample':           Real(0.6, 1.0, prior='uniform'),
    'xgb__colsample_bytree':    Real(0.6, 1.0, prior='uniform'),
}

bayes_search = BayesSearchCV(
    pipeline,
    search_space,
    n_iter=50,          # number of Bayesian optimization iterations
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
bayes_search.fit(X_train, y_train)

best_model = bayes_search.best_estimator_
print(f"\nBest Parameters Found: {bayes_search.best_params_}")
print(f"Best CV F1 Score:      {bayes_search.best_score_:.4f}")

print("\n4. Generating Predictions...")
train_pred = best_model.predict(X_train)
test_pred  = best_model.predict(X_test)

print("\n=== METRICS ===")
print(f"Training Balanced Accuracy: {balanced_accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Balanced Accuracy:  {balanced_accuracy_score(y_test, test_pred) * 100:.2f}%")
print(f"Training F1 Score:          {f1_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing F1 Score:           {f1_score(y_test, test_pred) * 100:.2f}%")
print(f"Training Accuracy (Std):    {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Accuracy (Std):     {accuracy_score(y_test, test_pred) * 100:.2f}%")
print(f"Training Precision:         {precision_score(y_train, train_pred) * 100:.2f}%")
print(f"Training Recall:            {recall_score(y_train, train_pred) * 100:.2f}%")
print(f"Testing Precision:          {precision_score(y_test, test_pred) * 100:.2f}%")
print(f"Testing Recall:             {recall_score(y_test, test_pred) * 100:.2f}%")

print("\n5. Saving QGIS CSV...")
test_df = df.loc[X_test.index].copy()
test_df['Predicted_Hotspot'] = test_pred

def categorize_prediction(row):
    if row['Is_Hotspot'] == 1 and row['Predicted_Hotspot'] == 1: return 'True Hotspot'
    if row['Is_Hotspot'] == 0 and row['Predicted_Hotspot'] == 0: return 'True Safe'
    if row['Is_Hotspot'] == 0 and row['Predicted_Hotspot'] == 1: return 'False Alarm'
    if row['Is_Hotspot'] == 1 and row['Predicted_Hotspot'] == 0: return 'Missed Hotspot'

test_df['Prediction_Category'] = test_df.apply(categorize_prediction, axis=1)
test_df.to_csv('QGIS_Predictions_BO_SMOTE_Tuned_XGB.csv', index=False)
print("Saved -> 'QGIS_Predictions_BO_SMOTE_Tuned_XGB.csv'")

print("\n6. Saving Confusion Matrix Image...")
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Predicted Safe", "Predicted Hotspot"],
            yticklabels=["Actually Safe", "Actually Hotspot"])
plt.title('Bayesian Optimized SMOTE + Tuned XGB - Confusion Matrix')
plt.ylabel('True Reality (Map)')
plt.xlabel('Algorithm Prediction')
plt.tight_layout()
plt.savefig('Matrix_BO_SMOTE_Tuned_XGB.png', dpi=300)
print("Saved -> 'Matrix_BO_SMOTE_Tuned_XGB.png'")

# 7. FEATURE IMPORTANCE BAR CHART
print("\n7. Saving Feature Importance Chart...")
xgb_model = best_model.named_steps['xgb']
feature_names = X.columns
importances = xgb_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#1565C0')
plt.xlabel('Importance')
plt.title('Bayesian Optimized SMOTE + XGBoost - Feature Importance')
plt.tight_layout()
plt.savefig('Feature_Importance_BO_SMOTE_XGB.png', dpi=300)
print("Saved -> 'Feature_Importance_BO_SMOTE_XGB.png'")
