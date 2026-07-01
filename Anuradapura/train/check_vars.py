import os
for f in os.listdir('.'):
    if f.endswith('.py'):
        with open(f, 'r') as file:
            content = file.read()
            model_var = 'UNKNOWN'
            if 'best_xgb = ' in content: model_var = 'best_xgb'
            elif 'best_rf = ' in content: model_var = 'best_rf'
            elif 'xgb = ' in content: model_var = 'xgb'
            elif 'rf = ' in content: model_var = 'rf'
            print(f'{f}: {model_var}')
