import os
import re
import subprocess
import json

py_files = [
    'random_forest.py',
    'random_forest_with_smote_data.py',
    'tuned_random_forest.py',
    'tuned_random_forest_with_bo_smote.py',
    'tuned_random_forest_with_smote.py',
    'tuned_xgboost.py',
    'tuned_xgboost_with_bo_smote.py',
    'tuned_xgboost_with_smote.py',
    'xgboost_basic.py',
    'xgboost_with_smote_data.py'
]

results = []

for f in py_files:
    if not os.path.exists(f): continue
    
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Disable plotting to prevent blocking
    content = re.sub(r'plt\.show\(\)', 'pass', content)
    content = re.sub(r'plt\.savefig\(.*\)', 'pass', content)
    content = content.replace('✅', '')
    
    injection = """
import json
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

_model = None
if 'best_model' in locals(): _model = best_model
elif 'best_xgb' in locals(): _model = best_xgb
elif 'best_rf' in locals(): _model = best_rf
elif 'xgb' in locals(): _model = xgb
elif 'rf' in locals(): _model = rf
elif 'rf_smote' in locals(): _model = rf_smote
elif 'xgb_smote' in locals(): _model = xgb_smote
elif 'pipeline' in locals(): _model = pipeline

if _model is not None:
    try:
        _train_pred = _model.predict(X_train)
        _test_pred = _model.predict(X_test)
        
        _metrics = {
            'Balanced_Accuracy': balanced_accuracy_score(y_test, _test_pred),
            'Testing_accuracy': accuracy_score(y_test, _test_pred),
            'Training_accuracy': accuracy_score(y_train, _train_pred),
            'precision': precision_score(y_test, _test_pred),
            'recall': recall_score(y_test, _test_pred),
            'F1_score': f1_score(y_test, _test_pred)
        }
        print("___JSON_START___" + json.dumps(_metrics) + "___JSON_END___")
    except Exception as e:
        print("___JSON_START___" + json.dumps({"Error": str(e)}) + "___JSON_END___")
"""
    
    temp_file = f"temp_{f}"
    with open(temp_file, 'w', encoding='utf-8') as file:
        file.write(content + "\n" + injection)
    
    python_exe = r"D:/Research/Train2/Kiribathgoda/Kiribathgoda/.venv/Scripts/python.exe"
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    print(f"Running {f}...")
    try:
        res = subprocess.run([python_exe, temp_file], capture_output=True, text=True, timeout=900, env=env)
        out = res.stdout
        
        match = re.search(r'___JSON_START___(.*?)___JSON_END___', out)
        if match:
            m = json.loads(match.group(1))
            m['File'] = f
            results.append(m)
        else:
            print(f"Failed to extract JSON for {f}.")
            print(f"Stderr: {res.stderr}")
            print(f"Stdout (last 500 chars): {out[-500:]}")
            results.append({'File': f, 'Error': 'Failed'})
    except subprocess.TimeoutExpired:
        print(f"Timeout for {f}")
        results.append({'File': f, 'Error': 'Timeout'})
    
    os.remove(temp_file)

# Now print as markdown table
print("| Model File | Balanced Accuracy | Testing Accuracy | Training Accuracy | Precision | Recall | F1 Score |")
print("|---|---|---|---|---|---|---|")
for r in results:
    if 'Error' in r:
        print(f"| {r.get('File', 'Unknown')} | Error | Error | Error | Error | Error | Error |")
    else:
        print(f"| {r['File']} | {r['Balanced_Accuracy']*100:.2f}% | {r['Testing_accuracy']*100:.2f}% | {r['Training_accuracy']*100:.2f}% | {r['precision']*100:.2f}% | {r['recall']*100:.2f}% | {r['F1_score']*100:.2f}% |")

