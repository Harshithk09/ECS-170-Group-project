'''
script_gcn_node_classification.py
===================================
Trains a 2-layer GCN on Cora, Citeseer, and Pubmed.

Run from PyCharm with working directory set to:
  ECS170_Spring_2026_Source_Code_Template/script/stage_5_script

Make sure the project root is marked as Sources Root in PyCharm
so that  `from code.base_class...`  and  `from local_code...`  resolve correctly.
'''

import sys
import os

# ----------------------------------------------------------
# Add the template root to path so all imports resolve
# (only needed if running outside PyCharm's Sources Root)
# ----------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN_Node_Classification     import Method_GCN_Node_Classification
from local_code.stage_5_code.Evaluate_Accuracy                  import Evaluate_Accuracy

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
DATA_ROOT     = '../../data/stage_5_data'
RESULT_FOLDER = '../../result/stage_5_result'

DATASETS = ['cora', 'citeseer', 'pubmed']

all_results = {}

for dataset_name in DATASETS:
    print('\n' + '='*60)
    print('Dataset: {}'.format(dataset_name.upper()))
    print('='*60)

    # ---- dataset loader ----
    loader = Dataset_Loader(dName=dataset_name, dDescription='')
    loader.dataset_name               = dataset_name
    loader.dataset_source_folder_path = os.path.join(DATA_ROOT, dataset_name)

    loaded         = loader.load()
    graph          = loaded['graph']
    train_test_val = loaded['train_test_val']

    # ---- method ----
    gcn_method = Method_GCN_Node_Classification(
        mName='GCN',
        mDescription='2-layer Graph Convolutional Network for node classification'
    )
    gcn_method.dataset_name  = dataset_name
    gcn_method.result_folder = RESULT_FOLDER
    gcn_method.data = {
        'X':         graph['X'],
        'y':         graph['y'],
        'utility':   graph['utility'],
        'idx_train': train_test_val['idx_train'],
        'idx_val':   train_test_val['idx_val'],
        'idx_test':  train_test_val['idx_test'],
    }

    pred_y, true_y = gcn_method.run()

    # ---- evaluate ----
    evaluator = Evaluate_Accuracy()
    metrics   = evaluator.evaluate(pred_y, true_y)

    print('\n--- {} Results ---'.format(dataset_name.upper()))
    print('Accuracy       : {:.4f}'.format(metrics['accuracy']))
    print('Macro F1       : {:.4f}'.format(metrics['f1_macro']))
    print('Weighted F1    : {:.4f}'.format(metrics['f1_weighted']))
    print('Macro Precision: {:.4f}'.format(metrics['precision_macro']))
    print('Macro Recall   : {:.4f}'.format(metrics['recall_macro']))
    print(metrics['report'])

    all_results[dataset_name] = metrics

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print('{:<12} {:>10} {:>10} {:>12}'.format('Dataset', 'Accuracy', 'Macro F1', 'Weighted F1'))
print('-'*48)
for name, r in all_results.items():
    print('{:<12} {:>10.4f} {:>10.4f} {:>12.4f}'.format(
        name, r['accuracy'], r['f1_macro'], r['f1_weighted']))
