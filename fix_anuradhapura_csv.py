import subprocess
import re
import pandas as pd
import os
import sys

csv_path = r"d:\Research\Train2\final_metrics_summary.csv"
df = pd.read_csv(csv_path)

anuradhapura_scripts = [
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

cwd = r"d:\Research\Train2\Anuradapura\Anuradapura\train"
python_exe = r"d:\Research\Train2\Anuradapura\.venv\Scripts\python.exe"
env = os.environ.copy()
env["PYTHONPATH"] = r"d:\Research\Train2"

results = []

for script in anuradhapura_scripts:
    cmd = [python_exe, "-c", f"import mock_matplotlib; import runpy; runpy.run_path('{script}')"]
    print(f"Running {script}...")
    try:
        out = subprocess.check_output(cmd, cwd=cwd, env=env, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError as e:
        out = e.output
        
    metrics = {}
    
    # Testing Balanced Accuracy
    m = re.search(r'Testing Balanced Accuracy:\s*([\d\.]+)%', out)
    if m: metrics['Balanced Accuracy (%)'] = float(m.group(1))
    
    # Testing Accuracy
    m = re.search(r'Testing Accuracy(?: \(Std\))?:\s*([\d\.]+)%', out)
    if m: metrics['Testing Accuracy (%)'] = float(m.group(1))
    
    # Training Accuracy
    m = re.search(r'Training Accuracy(?: \(Std\))?:\s*([\d\.]+)%', out)
    if m: metrics['Training Accuracy (%)'] = float(m.group(1))
    
    # Testing Precision
    m = re.search(r'Testing Precision:\s*([\d\.]+)%', out)
    if m: metrics['Precision (%)'] = float(m.group(1))
    
    # Testing Recall
    m = re.search(r'Testing Recall:\s*([\d\.]+)%', out)
    if m: metrics['Recall (%)'] = float(m.group(1))
    
    # Testing F1 Score
    m = re.search(r'Testing F1[- ]Score:\s*([\d\.]+)%', out)
    if m: metrics['F1 Score (%)'] = float(m.group(1))
    
    # Fallbacks for some scripts that don't prefix with "Testing"
    if 'Balanced Accuracy (%)' not in metrics:
        m = re.search(r'^Balanced Accuracy:\s*([\d\.]+)%', out, re.MULTILINE)
        if m: metrics['Balanced Accuracy (%)'] = float(m.group(1))
        
    if 'Testing Accuracy (%)' not in metrics:
        m = re.search(r'^Standard Accuracy:\s*([\d\.]+)%', out, re.MULTILINE)
        if m: metrics['Testing Accuracy (%)'] = float(m.group(1))
        
    if 'F1 Score (%)' not in metrics:
        m = re.search(r'^F1 Score:\s*([\d\.]+)%', out, re.MULTILINE)
        if m: metrics['F1 Score (%)'] = float(m.group(1))
        
    # Remove existing row in df
    df = df[~((df['Area'] == 'Anuradhapura') & (df['Model Script'] == script))]
    
    # Add new row
    new_row = {
        'Area': 'Anuradhapura',
        'Model Script': script,
        'Balanced Accuracy (%)': metrics.get('Balanced Accuracy (%)'),
        'Testing Accuracy (%)': metrics.get('Testing Accuracy (%)'),
        'Training Accuracy (%)': metrics.get('Training Accuracy (%)'),
        'Precision (%)': metrics.get('Precision (%)'),
        'Recall (%)': metrics.get('Recall (%)'),
        'F1 Score (%)': metrics.get('F1 Score (%)')
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
df = df.sort_values(by=['Area', 'Model Script'])
df.to_csv(csv_path, index=False)
print("Finished fixing Anuradhapura metrics in CSV.")
