'''
Stage 3 setting: train and test are already split in the pickle; load via dataset.load().
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Stage3_PreSplit(setting):
    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        self.method.data = loaded_data
        learned_result = self.method.run()
        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()
        self.result.data = {
            'metrics': {key: round(value, 4) for key, value in metrics.items()},
        }
        self.result.save()
        return metrics, None
