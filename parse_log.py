import re
import pandas as pd

log_path = r"C:\Users\OSADA\.gemini\antigravity-ide\brain\d4102432-f2bc-4cab-9d21-a058c323fdcf\.system_generated\tasks\task-189.log"

with open(log_path, 'r', encoding='utf-8') as f:
    log_content = f.read()

results = []
current_area = None
current_script = None

# We can find area transitions
for line in log_content.split('\n'):
    area_match = re.search(r'--- Running for Area: (.*?) ---', line)
    if area_match:
        current_area = area_match.group(1)
        continue
    
    script_match = re.search(r'Running (.*?) in .*?\.\.\.', line)
    if script_match:
        current_script = script_match.group(1)
        continue
        
# A more robust way: split by "Running <script> in <area>..."
chunks = re.split(r'Running (.*?) in (.*?)\.\.\.', log_content)

# chunks[0] is preamble.
# chunks[1] is script, chunks[2] is area, chunks[3] is output, etc.

results = []
for i in range(1, len(chunks), 3):
    f = chunks[i]
    area = chunks[i+1]
    out = chunks[i+2]
    
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

df = pd.DataFrame(results)
df.to_csv(r'd:\Research\Train2\all_areas_metrics_fixed.csv', index=False)
print("Saved to all_areas_metrics_fixed.csv")
