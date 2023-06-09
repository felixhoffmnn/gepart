from sklearn.metrics import accuracy_score, precision_recall_fscore_support

NUM_CLASSES = 6


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", labels=list(range(NUM_CLASSES))
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
