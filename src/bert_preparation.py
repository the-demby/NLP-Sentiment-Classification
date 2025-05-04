import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast

def prepare_bert_dataset(
    csv_path="Data/imdb_cleaned.csv",
    sample_size=None,
    tokenizer_name="distilbert-base-uncased",
    max_length=512
):
    df = pd.read_csv(csv_path)

    # Optionnel : échantillonnage équilibré
    if sample_size:
        df_pos = df[df['label'] == 1].sample(sample_size, random_state=42)
        df_neg = df[df['label'] == 0].sample(sample_size, random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Texte original ou nettoyé selon ton choix
    texts = df["cleaned_text"].tolist()
    labels = df["label"].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "label": labels
    })

    return dataset.train_test_split(test_size=0.2, seed=42)

def prepare_bert_dataset_with_er(csv_path="Data/imdb_cleaned.csv", sample_size=None):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if sample_size:
        df_pos = df[df.label == 1].sample(sample_size, random_state=42)
        df_neg = df[df.label == 0].sample(sample_size, random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(df["cleaned_text"].tolist(), truncation=True, padding=True, max_length=512)

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "label": df["label"].tolist(),
        "er_score": df["er_score"].tolist()  # ⬅️ on ajoute ici
    })

    return dataset.train_test_split(test_size=0.2, seed=42)