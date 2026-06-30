import re
import pandas as pd

log_path = r"C:\Users\OSADA\.gemini\antigravity-ide\brain\d4102432-f2bc-4cab-9d21-a058c323fdcf\.system_generated\tasks\task-189.log"

with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

results = []
current_area = None
current_script = None

metrics = {
    'Balanced Accuracy (%)': None,
    'Testing Accuracy (%)': None,
    'Training Accuracy (%)': None,
    'Precision (%)': None,
    'Recall (%)': None,
    'F1 Score (%)': None
}

for line in lines:
    area_match = re.search(r'--- Running for Area: (.*?) ---', line)
    if area_match:
        current_area = area_match.group(1)
        continue
    
    script_match = re.search(r'Running (.*?) in (.*)\.\.\.', line)
    if script_match:
        # If we had previous metrics, save them (unless empty)
        if current_script and any(v is not None for v in metrics.values()):
            results.append({
                'Area': current_area,
                'Model Script': current_script,
                **metrics
            })
            # Reset metrics
            metrics = {k: None for k in metrics.keys()}
            
        current_script = script_match.group(1)
        current_area = script_match.group(2)
        continue

    if current_script:
        # Parse metrics
        # Balanced Accuracy
        m = re.search(r'^(?:Testing )?Balanced Accuracy:\s*([\d\.]+)%', line.strip())
        if m: metrics['Balanced Accuracy (%)'] = float(m.group(1))
        
        # Testing Accuracy
        m = re.search(r'^Testing Accuracy(?: \(Std\))?:\s*([\d\.]+)%', line.strip())
        if m: metrics['Testing Accuracy (%)'] = float(m.group(1))
        
        # Training Accuracy
        m = re.search(r'^Training Accuracy(?: \(Std\))?:\s*([\d\.]+)%', line.strip())
        if m: metrics['Training Accuracy (%)'] = float(m.group(1))
        
        # Precision
        m = re.search(r'^(?:Testing )?Precision:\s*([\d\.]+)%', line.strip())
        if m: metrics['Precision (%)'] = float(m.group(1))
        
        # Recall
        m = re.search(r'^(?:Testing )?Recall:\s*([\d\.]+)%', line.strip())
        if m: metrics['Recall (%)'] = float(m.group(1))
        
        # F1 Score
        m = re.search(r'^(?:Testing )?F1[- ]Score:\s*([\d\.]+)%', line.strip())
        if m: metrics['F1 Score (%)'] = float(m.group(1))

# Append the final one
if current_script and any(v is not None for v in metrics.values()):
    results.append({
        'Area': current_area,
        'Model Script': current_script,
        **metrics
    })

df = pd.DataFrame(results)
# Sort by Area and Model Script
df = df.sort_values(by=['Area', 'Model Script'])
df.to_csv(r'd:\Research\Train2\final_all_areas_metrics.csv', index=False)

# Also generate a markdown table
markdown = "# Full ML Metrics Across All Areas\n\n"
for area in df['Area'].unique():
    area_df = df[df['Area'] == area]
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

with open(r'd:\Research\Train2\full_metrics.md', 'w', encoding='utf-8') as f:
    f.write(markdown)
    
print("Successfully generated final_all_areas_metrics.csv and full_metrics.md")
