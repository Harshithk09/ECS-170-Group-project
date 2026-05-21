'''
Concrete ResultModule class for Stage 2 experiment outputs.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from pathlib import Path
import pickle

from local_code.base_class.result import result


class Result_Saver(result):
    # The raw result payload produced by the method, usually a dictionary
    # containing predicted labels and true labels.
    data = None
    # Kept for compatibility with the Stage 1 API. Stage 2 usually saves a
    # single result file, so this can stay as None.
    fold_count = None
    # Directory where the pickle file should be written.
    result_destination_folder_path = None
    # Base filename to use for the saved pickle file.
    result_destination_file_name = None
    # Stores the exact path of the most recent saved file so load() can reopen it.
    last_saved_path = None

    def _build_destination_path(self):
        # Fail early if the caller forgot to configure the output location.
        if not self.result_destination_folder_path:
            raise ValueError('result_destination_folder_path must be set before save().')
        if not self.result_destination_file_name:
            raise ValueError('result_destination_file_name must be set before save().')

        # Create the destination directory automatically so save() works even
        # when stage_2_result does not already exist.
        destination_dir = Path(self.result_destination_folder_path)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Support filenames with or without an explicit extension.
        # Example:
        #   "MLP_prediction_result"   -> stem="MLP_prediction_result", suffix=".pkl"
        #   "MLP_prediction_result.pkl" -> stem="MLP_prediction_result", suffix=".pkl"
        base_name = Path(self.result_destination_file_name)
        stem = base_name.stem if base_name.suffix else base_name.name
        suffix = base_name.suffix if base_name.suffix else '.pkl'

        # Append the fold index only when one is actually provided. This keeps
        # Stage 2 filenames simple while still allowing reuse in CV-style setups.
        if self.fold_count is None:
            file_name = f'{stem}{suffix}'
        else:
            file_name = f'{stem}_{self.fold_count}{suffix}'

        return destination_dir / file_name

    def save(self):
        print('saving results...')
        # Build the final absolute output path and serialize self.data either as
        # a pickle or a human-readable text file depending on the extension.
        destination_path = self._build_destination_path()
        if destination_path.suffix.lower() == '.txt':
            with destination_path.open('w', encoding='utf-8') as result_file:
                result_file.write(self._to_text(self.data))
        else:
            with destination_path.open('wb') as result_file:
                pickle.dump(self.data, result_file)
        # Remember the exact saved path so load() can read back the same file.
        self.last_saved_path = str(destination_path)
        print('results saved to:', destination_path)
        return self.last_saved_path

    def _to_text(self, value, indent=0):
        prefix = ' ' * indent
        if isinstance(value, dict):
            lines = []
            for key, item in value.items():
                if isinstance(item, (dict, list, tuple)):
                    lines.append(f'{prefix}{key}:')
                    lines.append(self._to_text(item, indent + 2))
                else:
                    lines.append(f'{prefix}{key}: {item}')
            return '\n'.join(lines) + '\n'
        if isinstance(value, (list, tuple)):
            lines = [f'{prefix}- {item}' for item in value]
            return '\n'.join(lines) + '\n'
        return f'{prefix}{value}\n'

    def load(self):
        # Prefer reopening the most recently saved file. If save() has not been
        # called yet, rebuild the path from the configured folder/name settings.
        if self.last_saved_path is not None:
            destination_path = Path(self.last_saved_path)
        else:
            destination_path = self._build_destination_path()

        # Deserialize the saved payload and also store it back on self.data so
        # the object behaves like the original Stage 1 loader/saver pattern.
        if destination_path.suffix.lower() == '.txt':
            with destination_path.open('r', encoding='utf-8') as result_file:
                self.data = result_file.read()
        else:
            with destination_path.open('rb') as result_file:
                self.data = pickle.load(result_file)
        return self.data
