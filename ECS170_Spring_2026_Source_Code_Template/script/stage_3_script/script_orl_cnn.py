import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

loader = Dataset_Loader_ORL('ORL', '')
loader.dataset_source_folder_path = '../../data/stage_3_data/'
loader.dataset_source_file_name = 'ORL'

model = Method_CNN_ORL('CNN_ORL', '')
model.data = loader.load()
result = model.run()

evaluator = Evaluate_Accuracy('eval', '')
evaluator.data = {'true_y': result['true_y'], 'pred_y': result['pred_y']}
metrics = evaluator.evaluate()

print('\n===== ORL Test Results =====')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

print('Predicted:', result['pred_y'][:5])
print('True:     ', result['true_y'][:5])