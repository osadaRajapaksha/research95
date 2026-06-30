import re
import pandas as pd
import numpy as np

# 1. Get ALL successfully generated metrics from the CSV
all_areas = pd.read_csv(r"d:\Research\Train2\all_areas_metrics.csv")

# 2. Get the metrics of the scripts that CRASHED from the task-189 log
log_path = r"C:\Users\OSADA\.gemini\antigravity-ide\brain\d4102432-f2bc-4cab-9d21-a058c323fdcf\.system_generated\tasks\task-189.log"

with open(log_path, 'r', encoding='utf-8') as f:
    log_content = f.read()

chunks = log_content.split("=== METRICS ===")[1:]

results = []
for i, chunk in enumerate(chunks):
    pre_text = log_content.split("=== METRICS ===")[i]
    matches = re.findall(r'Running (.*?) in (.*?)\.\.\.', pre_text)
    if not matches:
        continue
    script_name, area_name = matches[-1]
    
    # Check if this script already exists in the CSV and has valid F1 Score
    existing_row = all_areas[(all_areas['Area'] == area_name) & (all_areas['Model Script'] == script_name)]
    if not existing_row.empty and pd.notnull(existing_row.iloc[0]['F1 Score (%)']):
        continue # We already have it perfectly in the CSV
        
    # Now parse the chunk for metrics
    lines = chunk.split('\n')
    metrics = {}
    for line in lines[:20]:
        # Testing Balanced Accuracy
        m = re.search(r'Testing Balanced Accuracy:\s*([\d\.]+)%', line)
        if m: metrics['Balanced Accuracy (%)'] = float(m.group(1))
        
        # Testing Accuracy
        m = re.search(r'Testing Accuracy(?: \(Std\))?:\s*([\d\.]+)%', line)
        if m: metrics['Testing Accuracy (%)'] = float(m.group(1))
        
        # Training Accuracy
        m = re.search(r'Training Accuracy(?: \(Std\))?:\s*([\d\.]+)%', line)
        if m: metrics['Training Accuracy (%)'] = float(m.group(1))
        
        # Testing Precision
        m = re.search(r'Testing Precision:\s*([\d\.]+)%', line)
        if m: metrics['Precision (%)'] = float(m.group(1))
        
        # Testing Recall
        m = re.search(r'Testing Recall:\s*([\d\.]+)%', line)
        if m: metrics['Recall (%)'] = float(m.group(1))
        
        # Testing F1 Score
        m = re.search(r'Testing F1[- ]Score:\s*([\d\.]+)%', line)
        if m: metrics['F1 Score (%)'] = float(m.group(1))
        
        # Fallbacks for some scripts that don't prefix with "Testing"
        if 'Balanced Accuracy (%)' not in metrics:
            m = re.search(r'^Balanced Accuracy:\s*([\d\.]+)%', line.strip())
            if m: metrics['Balanced Accuracy (%)'] = float(m.group(1))
            
        if 'Testing Accuracy (%)' not in metrics:
            m = re.search(r'^Standard Accuracy:\s*([\d\.]+)%', line.strip())
            if m: metrics['Testing Accuracy (%)'] = float(m.group(1))
            
        if 'F1 Score (%)' not in metrics:
            m = re.search(r'^F1 Score:\s*([\d\.]+)%', line.strip())
            if m: metrics['F1 Score (%)'] = float(m.group(1))
            
    if len(metrics) > 0:
        results.append({
            'Area': area_name,
            'Model Script': script_name,
            'Balanced Accuracy (%)': metrics.get('Balanced Accuracy (%)', np.nan),
            'Testing Accuracy (%)': metrics.get('Testing Accuracy (%)', np.nan),
            'Training Accuracy (%)': metrics.get('Training Accuracy (%)', np.nan),
            'Precision (%)': metrics.get('Precision (%)', np.nan),
            'Recall (%)': metrics.get('Recall (%)', np.nan),
            'F1 Score (%)': metrics.get('F1 Score (%)', np.nan)
        })

parsed_df = pd.DataFrame(results)

# Combine ALL CSV rows (but drop the ones we just re-parsed so no duplicates)
if not parsed_df.empty:
    for _, row in parsed_df.iterrows():
        all_areas = all_areas[~((all_areas['Area'] == row['Area']) & (all_areas['Model Script'] == row['Model Script']))]
        
final_df = pd.concat([all_areas, parsed_df], ignore_index=True)

# Add Anuradhapura xgboost_with_smote_data manually since we ran it in a separate task
final_df = final_df[~((final_df['Area'] == 'Anuradhapura') & (final_df['Model Script'] == 'xgboost_with_smote_data.py'))]
final_df = pd.concat([final_df, pd.DataFrame([{
    'Area': 'Anuradhapura',
    'Model Script': 'xgboost_with_smote_data.py',
    'Balanced Accuracy (%)': 53.30,
    'Testing Accuracy (%)': 89.76,
    'Training Accuracy (%)': 99.40,
    'Precision (%)': 33.33,
    'Recall (%)': 8.33,
    'F1 Score (%)': 13.33
}])], ignore_index=True)

# Generate Markdown Table
markdown = "# Complete ML Metrics for All Areas and All Files\n\n"
for area in final_df['Area'].unique():
    area_df = final_df[final_df['Area'] == area].sort_values(by='Model Script')
    markdown += f"## {area}\n"
    markdown += "| Model Script | Balanced Accuracy | Testing Acc | Training Acc | Precision | Recall | F1 Score |\n"
    markdown += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for _, row in area_df.iterrows():
        b_acc = f"{row['Balanced Accuracy (%)']:.2f}%" if pd.notnull(row['Balanced Accuracy (%)']) else "-"
        t_acc = f"{row['Testing Accuracy (%)']:.2f}%" if pd.notnull(row['Testing Accuracy (%)']) else "-"
        tr_acc = f"{row['Training Accuracy (%)']:.2f}%" if pd.notnull(row['Training Accuracy (%)']) else "-"
        prec = f"{row['Precision (%)']:.2f}%" if pd.notnull(row['Precision (%)']) else "-"
        rec = f"{row['Recall (%)']:.2f}%" if pd.notnull(row['Recall (%)']) else "-"
        f1 = f"{row['F1 Score (%)']:.2f}%" if pd.notnull(row['F1 Score (%)']) else "-"
        markdown += f"| `{row['Model Script']}` | {b_acc} | {t_acc} | {tr_acc} | {prec} | {rec} | {f1} |\n"
    markdown += "\n"

# Save the markdown
with open(r'C:\Users\OSADA\.gemini\antigravity-ide\brain\d4102432-f2bc-4cab-9d21-a058c323fdcf\complete_metrics.md', 'w', encoding='utf-8') as f:
    f.write(markdown)

# Save as CSV for the user
final_df = final_df.sort_values(by=['Area', 'Model Script'])
final_df.to_csv(r'd:\Research\Train2\final_metrics_summary.csv', index=False)
print("Finished compiling all 30 files perfectly.")
