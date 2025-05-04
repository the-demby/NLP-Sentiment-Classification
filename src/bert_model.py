import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DistilBertModel, DistilBertPreTrainedModel


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }

def train_bert_model(dataset, output_dir="./results", epochs=2, batch_size=32):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=output_dir + "/logs",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,                         # float16 sur GPU
        gradient_checkpointing=True,       
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer

def get_trainer(dataset, output_dir="./results", epochs=2, batch_size=32):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=output_dir + "/logs",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    return trainer

class DistilBertWithER(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim + 1, config.dim)  # ⬅️ +1 pour er_score
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.relu = nn.ReLU()

    def forward(self, input_ids=None, attention_mask=None, er_score=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = distilbert_output.last_hidden_state[:, 0]  # [batch_size, hidden_dim]

        # Assurer que er_score est bien au bon format
        if er_score is None:
            raise ValueError("Missing 'er_score' input")

        if len(er_score.shape) == 1:
            er_score = er_score.unsqueeze(1)  # [batch_size, 1]

        # Concat [CLS] + er_score
        combined = torch.cat((cls_output, er_score), dim=1)

        x = self.pre_classifier(combined)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class BERTDatasetWithER(Dataset):
    def __init__(self, dataset_hf):
        self.dataset = dataset_hf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "er_score": torch.tensor(item["er_score"], dtype=torch.float32),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }