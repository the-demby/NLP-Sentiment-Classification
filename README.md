# 🧐 NLP Sentiment Classification – IMDB Project

This project is part of the final evaluation for the NLP course. It investigates how classical and transformer-based models handle binary sentiment classification on the IMDB dataset, with a focus on the integration of lexicon-based polarity scores.

---

## 🗂️ Project Structure

```
NLP-Sentiment-Classification/
│
├── Data/                  # Raw and cleaned datasets (excluded from Git)
│   ├── imdb_cleaned.csv
│   └── svm_preds.npz
│
├── results/               # Model outputs (excluded from Git)
├── notebooks/             # Jupyter notebooks for each experiment phase
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── baseline_training.ipynb
│   ├── bert_finetuning.ipynb
│   └── comparative_analysis.ipynb
│
├── src/                   # All source code (modularized)
│   ├── utils/
│   │   ├── nltk_utils.py
│   │   └── __init__.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── baseline_model.py
│   ├── bert_model.py
│   ├── bert_preparation.py
│   └── er_lexicon.py
│
├── README.md
├── requirements.txt
├── install.sh
├── .gitignore
└── A_Comparative_Analysis_of_SVM_and_DistilBERT_with_Polarity_aware_Features.pdf
```

---

## ⚙️ Installation

You can install the environment in one line using the `install.sh` script:

```bash
bash install.sh
```

Alternatively, manually:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Project Phases

| Phase                            | Notebook                     |
| -------------------------------- | ---------------------------- |
| 🍊 Data exploration & statistics | `data_exploration.ipynb`     |
| 🧹 Text cleaning & ER scoring    | `preprocessing.ipynb`        |
| 📊 TF-IDF + SVM Baseline         | `baseline_training.ipynb`    |
| 🤖 DistilBERT fine-tuning        | `bert_finetuning.ipynb`      |
| 🔍 Error analysis & comparison   | `comparative_analysis.ipynb` |

---

## 🔬 Key Findings

* **DistilBERT** achieves **90.7% accuracy**, outperforming TF-IDF + SVM (89.3%)
* Adding `er_score` improves SVM slightly, but degrades BERT
* Embedding space visualizations show clear sentiment clustering
* Error analysis reveals that BERT handles subtle context well but underperforms on short, polarized reviews

---

## 📝 Final Report

The complete article, following the NeurIPS 2024 format, is included as a PDF:

📄 [`A_Comparative_Analysis_of_SVM_and_DistilBERT_with_Polarity_aware_Features.pdf`](./A_Comparative_Analysis_of_SVM_and_DistilBERT_with_Polarity_aware_Features.pdf)

---

## 👤 Author

**\[Ismael DEMBELE]** - MS DS
This project was conducted as part of the final evaluation for the \[ENSAE NLP Course, 2025].
# NLP-Sentiment-Classification