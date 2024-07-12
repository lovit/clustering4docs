import scipy.sparse as sp
import json
from pathlib import Path

def get_bow():
    folder = Path(__file__).parent
    x = sp.load_npz(folder / 'bow.npz')
    idx_to_vocab = json.load(open(folder / 'idx_to_vocab.json'))
    vocab_to_idx = {v: i for i, v in enumerate(idx_to_vocab)}
    return x, idx_to_vocab, vocab_to_idx
