import os
from loguru import logger

@logger.catch
def load_er_lexicon():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    vocab_path = os.path.join(base_dir, 'Data', 'aclImdb', 'imdb.vocab')
    er_path = os.path.join(base_dir, 'Data', 'aclImdb', 'imdbEr.txt')
    
    with open(vocab_path, encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]
    
    with open(er_path, encoding='utf-8') as f:
        scores = [float(line.strip()) for line in f.readlines()]
    
    return dict(zip(vocab, scores))

@logger.catch
def compute_mean_er_score(text, lexicon):
    tokens = text.split()  # Le texte est déjà nettoyé
    scores = [lexicon[word] for word in tokens if word in lexicon]
    return sum(scores) / len(scores) if scores else 0.0
