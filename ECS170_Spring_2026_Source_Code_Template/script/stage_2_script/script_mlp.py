from pathlib import Path
import sys

import numpy as np
import torch


# Resolve the project root from this script file so the script can be launched
# from any working directory and still find the local_code package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Insert the project root on sys.path so imports like
    # `from local_code.stage_2_code...` work reliably when running the script
    # directly with Python.
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split


def resolve_stage_2_data_paths(project_root):
    # Stage 2 data is expected to live under data/stage_2_data.
    data_dir = project_root / 'data' / 'stage_2_data'
    # Accept either the generic train/test naming or explicit MNIST filenames.
    candidate_pairs = [
        (data_dir / 'train.csv', data_dir / 'test.csv'),
        (data_dir / 'mnist_train.csv', data_dir / 'mnist_test.csv'),
    ]

    # Return the first matching pair found on disk.
    for train_path, test_path in candidate_pairs:
        if train_path.exists() and test_path.exists():
            return train_path, test_path

    # Build a detailed error message so the user knows exactly where to place
    # the dataset files and which filenames are accepted.
    expected_pairs = '\n'.join(
        f'  - {train_path.name} and {test_path.name}'
        for train_path, test_path in candidate_pairs
    )
    raise FileNotFoundError(
        'Could not find the Stage 2 MNIST CSV files.\n'
        f'Expected them under: {data_dir}\n'
        f'Accepted filename pairs:\n{expected_pairs}'
    )


def main():
    # Fix the random seeds so repeated runs are more reproducible.
    np.random.seed(2)
    torch.manual_seed(2)

    # Locate the Stage 2 MNIST CSV files before creating the experiment objects.
    train_path, test_path = resolve_stage_2_data_paths(PROJECT_ROOT)

    # Dataset_Loader is instantiated for consistency with the template's
    # dataset/method/setting/result/evaluate architecture, even though this
    # Stage 2 setting loads the CSV files directly from disk.
    data_obj = Dataset_Loader('mnist', 'MNIST classification dataset')
    # The MLP architecture itself is defined in stage_2_code/Method_MLP.py.
    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.plot_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_2_result')
    method_obj.plot_file_name = 'MLP_convergence_curve.png'

    # Configure where the raw prediction results should be saved.
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_2_result')
    result_obj.result_destination_file_name = 'MLP_prediction_result.pkl'

    # This setting expects two already-separated CSV files rather than doing
    # a random split like Stage 1.
    setting_obj = Setting_Train_Test_Split('train test split', '')
    # The evaluator reports accuracy plus multiclass F1/precision/recall.
    evaluate_obj = Evaluate_Accuracy('classification metrics', '')

    print('************ Start ************')
    print('Train CSV:', train_path)
    print('Test CSV :', test_path)

    # Wire the standard template objects together.
    # After prepare(), the setting object knows which dataset, method, result,
    # and evaluation components it should orchestrate.
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # Load the two CSV files into the setting object.
    setting_obj.load(str(train_path), str(test_path))
    setting_obj.print_setup_summary()

    # Run the full Stage 2 pipeline:
    #   1. split features/labels from the train and test CSV files
    #   2. train the MLP on the training data
    #   3. predict labels on the test data
    #   4. save the raw predictions
    #   5. compute the evaluation metrics
    metrics, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    # Print each metric returned by Evaluate_Accuracy in a readable format.
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')

    # The saver records the exact file path after a successful save.
    if result_obj.last_saved_path is not None:
        print('Saved result:', result_obj.last_saved_path)
    if method_obj.last_plot_path is not None:
        print('Saved plot  :', method_obj.last_plot_path)

    print('************ Finish ************')


if __name__ == '__main__':
    main()
