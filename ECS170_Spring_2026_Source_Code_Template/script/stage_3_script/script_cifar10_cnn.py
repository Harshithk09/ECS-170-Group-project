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
from local_code.stage_3_code.Dataset_Loader_CIFAR10 import Dataset_Loader_CIFAR10
from local_code.stage_3_code.Method_CNN_CIFAR10 import Method_CNN_CIFAR10
from local_code.stage_3_code.Setting_Stage3_PreSplit import Setting_Stage3_PreSplit


def resolve_cifar10_pickle(project_root):
    data_dir = project_root / 'data' / 'stage_3_data'
    candidates = [data_dir / 'CIFAR', data_dir / 'CIFAR.pkl', data_dir / 'cifar10.pkl']
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        'Could not find CIFAR-10 pickle. Place the course file as '
        f'{data_dir / "CIFAR"} (or CIFAR.pkl) per script_data_loader.py.'
    )


def main():
    np.random.seed(2)
    torch.manual_seed(2)
    random.seed(2)

    data_dir = PROJECT_ROOT / 'data' / 'stage_3_data'
    data_obj = Dataset_Loader_CIFAR10('cifar10', 'CIFAR-10 image classification')
    pickle_path = None
    try:
        pickle_path = resolve_cifar10_pickle(PROJECT_ROOT)
        data_obj.dataset_source_folder_path = str(pickle_path.parent) + '/'
        data_obj.dataset_source_file_name = pickle_path.name
    except FileNotFoundError:
        tv_root = data_dir / 'cifar10_official'
        print('No course pickle found; downloading CIFAR-10 via torchvision ->', tv_root)
        data_obj.preloaded_data = Dataset_Loader_CIFAR10.course_dict_from_torchvision(tv_root)

    method_obj = Method_CNN_CIFAR10('cnn cifar10', 'WideResNet-28-10 on CIFAR-10')
    method_obj.architecture = 'resnet18'
    method_obj.max_epoch = 50
    method_obj.plot_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_3_result')
    method_obj.plot_file_name = 'cifar10_convergence_curve.png'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_3_result')
    result_obj.result_destination_file_name = 'CIFAR10_CNN_metrics.txt'

    setting_obj = Setting_Stage3_PreSplit('pre-split train test', '')
    evaluate_obj = Evaluate_Accuracy('classification metrics', '')

    print('************ Start ************')
    if pickle_path is not None:
        print('CIFAR pickle:', pickle_path)
    else:
        print('CIFAR-10: torchvision official split (same 50k/10k as the assignment)')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    metrics, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')

    if result_obj.last_saved_path is not None:
        print('Saved result:', result_obj.last_saved_path)
    if method_obj.last_plot_path is not None:
        print('Saved plot  :', method_obj.last_plot_path)

    print('************ Finish ************')


if __name__ == '__main__':
    main()
