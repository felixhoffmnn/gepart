import torch
import torch.backends.mps
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Sentiment:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_name = "oliverguhr/german-sentiment-bert"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def predict_sentiment(self, text: str):
        encoded = self.tokenizer.encode_plus(
            text, padding=True, add_special_tokens=True, truncation=True, return_tensors="pt"
        )
        encoded = encoded.to(self.device)
        with torch.no_grad():
            max_pos = self.model(**encoded).logits.argmax().item()

        return self.model.config.id2label[max_pos]

    def predict_batch_sentiment(self, texts: list[str]):
        encoded = self.tokenizer.batch_encode_plus(
            texts, padding=True, add_special_tokens=True, truncation=True, return_tensors="pt"
        )
        encoded = encoded.to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded).logits

        return [self.model.config.id2label[element.argmax().item()] for element in logits]
