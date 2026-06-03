'''
Evaluation Metrics for Node Classification (Stage 5)
'''

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import numpy as np


class Evaluate_Metrics:

    def __init__(self):
        pass

    def evaluate(self, pred_y, true_y):
        """
        Compute accuracy, macro-F1, weighted-F1, precision, recall.

        Parameters
        ----------
        pred_y : torch.Tensor or np.ndarray  (1-D, integer class labels)
        true_y : torch.Tensor or np.ndarray  (1-D, integer class labels)

        Returns
        -------
        dict with metric names as keys
        """
        # to numpy
        if hasattr(pred_y, 'numpy'):
            pred_y = pred_y.numpy()
        if hasattr(true_y, 'numpy'):
            true_y = true_y.numpy()

        acc      = accuracy_score(true_y, pred_y)
        prec_mac = precision_score(true_y, pred_y, average='macro',    zero_division=0)
        rec_mac  = recall_score(   true_y, pred_y, average='macro',    zero_division=0)
        f1_mac   = f1_score(       true_y, pred_y, average='macro',    zero_division=0)
        f1_wt    = f1_score(       true_y, pred_y, average='weighted', zero_division=0)

        report = classification_report(true_y, pred_y, zero_division=0)

        results = {
            'accuracy':          acc,
            'macro_precision':   prec_mac,
            'macro_recall':      rec_mac,
            'macro_f1':          f1_mac,
            'weighted_f1':       f1_wt,
            'classification_report': report,
        }
        return results

    def print_results(self, results, dataset_name=''):
        header = f'Evaluation Results – {dataset_name}' if dataset_name else 'Evaluation Results'
        print(f'\n{"="*60}')
        print(f'  {header}')
        print(f'{"="*60}')
        print(f'  Accuracy         : {results["accuracy"]:.4f}')
        print(f'  Macro Precision  : {results["macro_precision"]:.4f}')
        print(f'  Macro Recall     : {results["macro_recall"]:.4f}')
        print(f'  Macro F1         : {results["macro_f1"]:.4f}')
        print(f'  Weighted F1      : {results["weighted_f1"]:.4f}')
        print(f'\n  Per-class Report:')
        print(results['classification_report'])
