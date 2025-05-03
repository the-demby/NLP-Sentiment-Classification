import os
import nltk

# Utilisation du dossier local
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path = [NLTK_DATA_PATH]

def download_nltk_resources():
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
    nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)