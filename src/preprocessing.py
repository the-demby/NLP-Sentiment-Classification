import re
import os
import string
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from .utils.nltk_utils import download_nltk_resources

# Assure qu'on a tout ce qu'il faut
download_nltk_resources()

stop_words = set(stopwords.words('english') + list(string.punctuation))

@logger.catch
def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)        # balises <br>
    text = re.sub(r'<.*?>', ' ', text)            # autres balises HTML
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(cleaned)