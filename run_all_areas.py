import subprocess
import re
import os
import pandas as pd

directories = {
    'Ella': (r'd:\Research\Train2\Train2', r'd:\Research\Train2\Train2\.venv\Scripts\python.exe'),
    'Anuradhapura': (r'd:\Research\Train2\Anuradapura\Anuradapura\train', r'd:\Research\Train2\Anuradapura\.venv\Scripts\python.exe'),
    'Kiribathgoda': (r'd:\Research\Train2\Kiribathgoda\Kiribathgoda\train', r'd:\Research\Train2\Kiribathgoda\Kiribathgoda\.venv\Scripts\python.exe')
}

files = [
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
os.environ['PYTHONPATH'] = r'd:\Research\Train2'

for area, (directory, python_exe) in directories.items():
    print(f"\n--- Running for Area: {area} ---")
    for f in files:
        filepath = os.path.join(directory, f)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"Running {f} in {area}...")
        try:
            cmd = [python_exe, '-c', f"import mock_matplotlib; import runpy; runpy.run_path('{f}')"]
            out = subprocess.check_output(cmd, cwd=directory, stderr=subprocess.STDOUT, text=True)
            # Parse metrics
            train_acc = re.search(r'Training Accuracy:\s*([\d\.]+)%', out)
            test_acc = re.search(r'Testing Accuracy:\s*([\d\.]+)%', out)
            
            bal_acc = re.search(r'Balanced Accuracy:\s*([\d\.]+)%', out)
            if not bal_acc:
                bal_acc = re.search(r'Testing Balanced Accuracy:\s*([\d\.]+)%', out)
                
            prec = re.search(r'Precision:\s*([\d\.]+)%', out)
            if not prec:
                prec = re.search(r'Testing Precision:\s*([\d\.]+)%', out)
                
            rec = re.search(r'Recall:\s*([\d\.]+)%', out)
            if not rec:
                rec = re.search(r'Testing Recall:\s*([\d\.]+)%', out)
                
            f1 = re.search(r'F1[- ]Score:\s*([\d\.]+)%', out)
            if not f1:
                f1 = re.search(r'Testing F1 Score:\s*([\d\.]+)%', out)
            
            results.append({
                'Area': area,
                'Model Script': f,
                'Balanced Accuracy (%)': float(bal_acc.group(1)) if bal_acc else None,
                'Testing Accuracy (%)': float(test_acc.group(1)) if test_acc else None,
                'Training Accuracy (%)': float(train_acc.group(1)) if train_acc else None,
                'Precision (%)': float(prec.group(1)) if prec else None,
                'Recall (%)': float(rec.group(1)) if rec else None,
                'F1 Score (%)': float(f1.group(1)) if f1 else None,
            })
        except subprocess.CalledProcessError as e:
            print(f"Error running {f} in {area}:\n{e.output}")

df = pd.DataFrame(results)
df.to_csv(r'd:\Research\Train2\all_areas_metrics.csv', index=False)
print("Finished saving to all_areas_metrics.csv")
