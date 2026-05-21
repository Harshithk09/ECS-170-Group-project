'''
Concrete Evaluate class for classification metrics (accuracy, F1, precision, recall).
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import importlib
import numpy as np

from local_code.base_class.evaluate import evaluate


def _sklearn_metrics():
    """Defer sklearn.metrics import so static analysis succeeds without resolving scikit-learn."""
    sm = importlib.import_module('sklearn.metrics')
    return sm.accuracy_score, sm.f1_score, sm.precision_score, sm.recall_score


def _to_numpy_labels(y):
    # Avoid importing torch at module level so static analysis works when PyTorch
    # is not installed in the IDE's selected environment; torch tensors still convert.
    if hasattr(y, 'detach') and hasattr(y, 'cpu') and hasattr(y, 'numpy'):
        y = y.detach().cpu().numpy()
    y = np.asarray(y).ravel()
    return y


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        if self.data is None or 'true_y' not in self.data or 'pred_y' not in self.data:
            raise ValueError('data must be set with keys true_y and pred_y before evaluate().')

        y_true = _to_numpy_labels(self.data['true_y'])
        y_pred = _to_numpy_labels(self.data['pred_y'])

        accuracy_score, f1_score, precision_score, recall_score = _sklearn_metrics()
        out = {'accuracy': accuracy_score(y_true, y_pred)}
        for avg in ('weighted', 'macro', 'micro'):
            out[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
            out[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            out[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        return out

    def __str__(self):
        m = self.evaluate()
        lines = [
            f"Accuracy : {m['accuracy']:.4f}",
            f"F1 (weighted/macro/micro): {m['f1_weighted']:.4f} / {m['f1_macro']:.4f} / {m['f1_micro']:.4f}",
            f"Precision (weighted/macro/micro): {m['precision_weighted']:.4f} / {m['precision_macro']:.4f} / {m['precision_micro']:.4f}",
            f"Recall (weighted/macro/micro): {m['recall_weighted']:.4f} / {m['recall_macro']:.4f} / {m['recall_micro']:.4f}",
        ]
        return '\n'.join(lines)
