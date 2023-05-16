from pathlib import Path

import pandas as pd
import pyarrow as pa
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (  # EarlyStoppingCallback,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def process_data(row):
    text = row["clean_text"]
    text = " ".join(text.split())

    encodings = tokenizer(text, truncation=True, padding=True, max_length=512)

    label = row["party"]

    encodings["label"] = label
    encodings["text"] = text

    return encodings


def compute_metrics(pred):
    labels = pred.label_ids
    print(labels)
    preds = pred.predictions.argmax(-1)
    print(preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", labels=[0, 1, 2, 3, 4, 5]
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


checkpoint = "bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
# Other possible models:
# bert-base-german-cased
# dbmdz/bert-base-german-uncased
# deepset/gbert-base
# german-nlp-group/electra-base-german-uncased
# distilbert-base-german-cased


data_dir_pp = Path("../../data/party_programs")
df_pp = pd.read_parquet(data_dir_pp / "party_programs.parquet")

data_dir_speeches = Path("../../data/speeches")
df_speeches = pd.read_parquet(data_dir_speeches / "speeches.parquet")

df_sample = pd.concat([df_pp, df_speeches], ignore_index=True).sample(1000, random_state=42)


processed_data = [process_data(row) for _, row in tqdm(df_sample.iterrows())]
df_processed = pd.DataFrame(processed_data)

df_train, df_test = train_test_split(df_processed, test_size=0.2, random_state=42)
train_hg = Dataset(pa.Table.from_pandas(df_train))
test_hg = Dataset(pa.Table.from_pandas(df_test))


training_args = TrainingArguments(
    output_dir="out",
    overwrite_output_dir=True,
    num_train_epochs=1,
    use_mps_device=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=0.001,
    do_train=True,
    do_eval=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hg,
    eval_dataset=test_hg,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.evaluate(test_hg)
