from pathlib import Path
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Dataset_Loader_Classification import load_classification_data
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification


def run_classification(cell_type: str):
    """
    Train and evaluate one RNN cell type for sentiment classification.
    Called three times from main() for RNN, LSTM, and GRU.
    """
    print(f'\n============================================================')
    print(f'  Cell type: {cell_type.upper()}')
    print(f'============================================================')

    # ── 1. Load data ──────────────────────────────────────────
    data_root = str(PROJECT_ROOT / 'data' / 'stage_4_data' / 'text_classification')
    train_loader, test_loader, vocab, label2id = load_classification_data(
        data_root      = data_root,
        max_vocab_size = 20000,
        max_seq_len    = 200,
        batch_size     = 64,
    )

    # ── 2. Build method object ────────────────────────────────
    method_obj = Method_RNN_Classification(
        f'rnn_classification_{cell_type}',
        f'RNN sentiment classification — {cell_type.upper()}'
    )
    method_obj.cell_type    = cell_type
    method_obj.vocab_size   = len(vocab)
    method_obj.num_classes  = len(label2id)
    method_obj.max_epoch    = 10
    method_obj.plot_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_4_result')
    method_obj.plot_file_name = f'RNN_classification_convergence_{cell_type}.png'

    # Re-initialise the layers now that vocab_size and num_classes are set
    method_obj.__init__(
        f'rnn_classification_{cell_type}',
        f'RNN sentiment classification — {cell_type.upper()}'
    )
    method_obj.cell_type    = cell_type
    method_obj.vocab_size   = len(vocab)
    method_obj.num_classes  = len(label2id)
    method_obj.plot_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_4_result')
    method_obj.plot_file_name = f'RNN_classification_convergence_{cell_type}.png'

    # Pass data loaders via the .data dict (same pattern as other stages)
    method_obj.data = {
        'train_loader': train_loader,
        'test_loader':  test_loader,
    }

    # ── 3. Result saver ───────────────────────────────────────
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_4_result')
    result_obj.result_destination_file_name   = f'RNN_classification_metrics_{cell_type}.txt'

    # ── 4. Train + test ───────────────────────────────────────
    print('--start training...')
    results = method_obj.run()

    # ── 5. Evaluate ───────────────────────────────────────────
    evaluate_obj = Evaluate_Accuracy('classification metrics', '')
    evaluate_obj.data = {
        'true_y': results['true_y'],
        'pred_y': results['pred_y'],
    }
    metrics = evaluate_obj.evaluate()

    # ── 6. Save results to text file ──────────────────────────
    result_dir = Path(result_obj.result_destination_folder_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / result_obj.result_destination_file_name

    with open(result_path, 'w') as f:
        f.write(f'Cell type : {cell_type.upper()}\n')
        f.write(f'Epochs    : {method_obj.max_epoch}\n')
        f.write(f'Vocab size: {len(vocab)}\n\n')
        f.write('---- Evaluation Metrics ----\n')
        for name, value in metrics.items():
            f.write(f'{name}: {value:.4f}\n')

    print(f'\n---- {cell_type.upper()} Results ----')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')
    print(f'Metrics saved to : {result_path}')
    print(f'Plot saved to    : {results["convergence_plot_path"]}')

    return metrics


def main():
    np.random.seed(2)
    torch.manual_seed(2)
    random.seed(2)

    print('************ Stage 4: RNN Text Classification ************')

    # Run all three cell types — covers parts 4-2, 4-3, and 4-5
    summary = {}
    for cell_type in ['rnn', 'lstm', 'gru']:
        summary[cell_type] = run_classification(cell_type)

    # ── Final comparison table ────────────────────────────────
    print('\n************ Overall Comparison ************')
    print(f'{"Cell":<6}  {"Accuracy":>10}  {"F1 (weighted)":>14}')
    print('-' * 36)
    for cell, metrics in summary.items():
        print(f'{cell.upper():<6}  '
              f'{metrics["accuracy"]:>10.4f}  '
              f'{metrics["f1_weighted"]:>14.4f}')

    print('\n************ Finish ************')


if __name__ == '__main__':
    main()