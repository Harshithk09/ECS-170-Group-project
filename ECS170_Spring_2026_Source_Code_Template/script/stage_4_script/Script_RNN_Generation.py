from pathlib import Path
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Dataset_Loader_Generation import load_generation_data
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation


# ── Starting words for generation (edit these for your report examples) ──
# The README says to use three words that exist in the dataset.
# 'what did the' is the example given in the ReadMe.docx
START_WORDS = ['what', 'did', 'the']


def run_generation(cell_type: str, train_loader, vocab, id2word, corpus_text):
    """
    Train one RNN cell type and generate a joke from START_WORDS.
    Called three times from main() for RNN, LSTM, and GRU.
    """
    print(f'\n============================================================')
    print(f'  Cell type: {cell_type.upper()}')
    print(f'============================================================')

    # ── 1. Build method object ────────────────────────────────
    method_obj = Method_RNN_Generation(
        f'rnn_generation_{cell_type}',
        f'RNN joke generation — {cell_type.upper()}'
    )
    method_obj.cell_type    = cell_type
    method_obj.vocab_size   = len(vocab)
    method_obj.max_epoch    = 20
    method_obj.start_words  = START_WORDS
    method_obj.plot_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_4_result')
    method_obj.plot_file_name = f'RNN_generation_convergence_{cell_type}.png'

    # Re-initialise layers now that vocab_size is set
    method_obj.__init__(
        f'rnn_generation_{cell_type}',
        f'RNN joke generation — {cell_type.upper()}'
    )
    method_obj.cell_type    = cell_type
    method_obj.vocab_size   = len(vocab)
    method_obj.plot_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_4_result')
    method_obj.plot_file_name = f'RNN_generation_convergence_{cell_type}.png'

    # Pass everything the model needs via .data dict
    method_obj.data = {
        'train_loader': train_loader,
        'start_words':  START_WORDS,
        'vocab':        vocab,
        'id2word':      id2word,
    }

    # ── 2. Result saver ───────────────────────────────────────
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_4_result')
    result_obj.result_destination_file_name   = f'RNN_generation_output_{cell_type}.txt'

    # ── 3. Train + generate ───────────────────────────────────
    results = method_obj.run()

    generated_text = results['generated_text']
    final_loss     = results['training_history']['loss'][-1]
    final_ppl      = results['training_history']['perplexity'][-1]

    # ── 4. Compare generated text against training corpus ─────
    # Count how many 3-word phrases in the generated text also
    # appear in the training corpus — rough fluency check
    def get_trigrams(text):
        words = text.lower().split()
        return set(tuple(words[i:i+3]) for i in range(len(words) - 2))

    gen_trigrams    = get_trigrams(generated_text)
    corpus_trigrams = get_trigrams(corpus_text)
    overlap = (len(gen_trigrams & corpus_trigrams) / len(gen_trigrams)
               if gen_trigrams else 0.0)

    # ── 5. Save results ───────────────────────────────────────
    result_dir = Path(result_obj.result_destination_folder_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / result_obj.result_destination_file_name

    with open(result_path, 'w') as f:
        f.write(f'Cell type    : {cell_type.upper()}\n')
        f.write(f'Epochs       : {method_obj.max_epoch}\n')
        f.write(f'Start words  : {" ".join(START_WORDS)}\n\n')
        f.write('---- Generated Text ----\n')
        f.write(generated_text + '\n\n')
        f.write('---- Evaluation ----\n')
        f.write(f'Final loss        : {final_loss:.4f}\n')
        f.write(f'Final perplexity  : {final_ppl:.2f}\n')
        f.write(f'3-gram overlap    : {overlap:.4f}\n')

    print(f'\n---- {cell_type.upper()} Generated Text ----')
    print(generated_text)
    print(f'\nFinal loss       : {final_loss:.4f}')
    print(f'Final perplexity : {final_ppl:.2f}')
    print(f'3-gram overlap   : {overlap:.4f}')
    print(f'Output saved to  : {result_path}')
    print(f'Plot saved to    : {results["convergence_plot_path"]}')

    return {
        'generated_text': generated_text,
        'final_loss':     final_loss,
        'final_ppl':      final_ppl,
        'overlap':        overlap,
    }


def main():
    np.random.seed(2)
    torch.manual_seed(2)
    random.seed(2)

    print('************ Stage 4: RNN Text Generation ************')
    print(f'Start words: {START_WORDS}')

    # ── Load data once — shared across all three cell types ───
    data_path = str(PROJECT_ROOT / 'data' / 'stage_4_data' / 'text_generation' / 'data')
    train_loader, vocab, id2word, corpus_text = load_generation_data(
        data_path      = data_path,
        seq_len        = 20,
        max_vocab_size = 8000,
        batch_size     = 64,
    )

    # ── Run all three cell types — covers parts 4-4 and 4-5 ──
    summary = {}
    for cell_type in ['rnn', 'lstm', 'gru']:
        summary[cell_type] = run_generation(
            cell_type, train_loader, vocab, id2word, corpus_text)

    # ── Final comparison table ────────────────────────────────
    print('\n************ Overall Comparison ************')
    print(f'{"Cell":<6}  {"Final Loss":>12}  {"Perplexity":>12}  {"3-gram Overlap":>16}')
    print('-' * 52)
    for cell, m in summary.items():
        print(f'{cell.upper():<6}  '
              f'{m["final_loss"]:>12.4f}  '
              f'{m["final_ppl"]:>12.2f}  '
              f'{m["overlap"]:>16.4f}')

    print('\n************ Finish ************')


if __name__ == '__main__':
    main()