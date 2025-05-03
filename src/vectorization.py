import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

def vectorize_text(df, use_er_score=False, max_features=20000):
    # TF-IDF sur les textes nettoyés
    tfidf = TfidfVectorizer(max_features=max_features)
    X_tfidf = tfidf.fit_transform(df["cleaned_text"])
    
    # Optionnel : ajout du score ER comme feature numérique
    if use_er_score:
        scaler = StandardScaler()
        er_scaled = scaler.fit_transform(df["er_score"].values.reshape(-1, 1))
        X = hstack([X_tfidf, er_scaled])
    else:
        X = X_tfidf

    y = df["label"].values
    return X, y, tfidf