from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_2_code.Result_Saver import Result_Saver


def main():
    from pathlib import Path
    import sys

    import numpy as np
    import torch

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
    from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
    from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
    from local_code.stage_2_code.Result_Saver import Result_Saver

    def main():
        np.random.seed(2)
        torch.manual_seed(2)

        data_path = PROJECT_ROOT / 'data' / 'stage_3_data' / 'MNIST'

        if not data_path.exists():
            raise FileNotFoundError(
                f'Could not find MNIST dataset at: {data_path}'
            )

        data_obj = Dataset_Loader_MNIST('mnist', 'MNIST image classification dataset')
        data_obj.dataset_source_folder_path = str(PROJECT_ROOT / 'data' / 'stage_3_data') + '/'
        data_obj.dataset_source_file_name = 'MNIST'

        loaded_data = data_obj.load()

        method_obj = Method_CNN_MNIST('CNN MNIST', '')
        method_obj.data = {
            'train': {
                'X': loaded_data['train']['X'],
                'y': loaded_data['train']['y']
            },
            'test': {
                'X': loaded_data['test']['X'],
                'y': loaded_data['test']['y']
            }

        }

        method_obj.plot_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_3_result')
        method_obj.plot_file_name = 'CNN_MNIST_convergence_curve.png'

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_3_result')
        result_obj.result_destination_file_name = 'CNN_MNIST_metrics.txt'

        evaluate_obj = Evaluate_Accuracy('classification metrics', '')

        print('************ Start MNIST CNN ************')
        print('Dataset:', data_path)

        method_result = method_obj.run()

        evaluate_obj.data = {
            'true_y': method_result['true_y'],
            'pred_y': method_result['pred_y']
        }

        metrics = evaluate_obj.evaluate()

        result_dir = Path(result_obj.result_destination_folder_path)
        result_dir.mkdir(parents=True, exist_ok=True)

        result_path = result_dir / result_obj.result_destination_file_name

        with open(result_path, 'w') as f:
            f.write('metrics:\n')
            for metric_name, metric_value in metrics.items():
                f.write(f'    {metric_name}: {metric_value:.4f}\n')

        result_obj.last_saved_path = str(result_path)

        print('************ Overall Performance ************')
        for metric_name, metric_value in metrics.items():
            print(f'{metric_name}: {metric_value:.4f}')

        print('Saved result:', result_obj.last_saved_path)

        if method_obj.last_plot_path is not None:
            print('Saved plot  :', method_obj.last_plot_path)

        print('************ Finish MNIST CNN ************')

    if __name__ == '__main__':
        main()

if __name__ == '__main__':
    main()