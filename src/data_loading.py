from loguru import logger
import os
import pandas as pd

@logger.catch
def load_imdb_dataset(base_path='Data/aclImdb'):
    def parse_directory(path, label=None):
        data = []
        for sentiment in ['pos', 'neg']:
            folder = os.path.join(path, sentiment)
            if not os.path.exists(folder): continue
            files = os.listdir(folder)
            for fname in files:
                if not fname.endswith(".txt"): continue
                fpath = os.path.join(folder, fname)
                with open(fpath, encoding='utf-8') as f:
                    review = f.read()
                rating = int(fname.split("_")[1].split(".")[0])
                data.append({
                    "text": review,
                    "label": 1 if sentiment == "pos" else 0,
                    "rating": rating,
                    "split": label,
                    "file": fname
                })
        return pd.DataFrame(data)

    train_df = parse_directory(os.path.join(base_path, "train"), label="train")
    test_df = parse_directory(os.path.join(base_path, "test"), label="test")
    
    return pd.concat([train_df, test_df], ignore_index=True)