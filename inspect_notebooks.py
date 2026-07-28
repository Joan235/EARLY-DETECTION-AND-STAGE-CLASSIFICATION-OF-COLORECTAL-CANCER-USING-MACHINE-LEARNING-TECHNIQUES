import json
from pathlib import Path

root = Path.cwd()
for rel in ['Notebooks/CRC Cancer Detection.ipynb','Notebooks/CRC Stage Classification.ipynb']:
    print(f'\n=== {rel} ===')
    nb = json.loads((root/rel).read_text(encoding='utf-8'))
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        if any(k in src for k in ['TARGET =', 'LabelEncoder', 'StandardScaler', 'X_train', 'X_test', 'feature_names', 'CRC_Risk', 'SMOTE', 'Stage_at_Diagnosis', 'class_names']):
            print(f'--- cell {i} ---')
            print(src[:4000])
            print()
