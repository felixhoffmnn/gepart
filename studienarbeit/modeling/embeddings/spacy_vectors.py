from typing import Any

import spacy

from studienarbeit.modeling.embeddings.embedding import Embedding


class SpaCy(Embedding):
    def __init__(self, max_vocab_size: int, max_len: int, embedding_dim: int):
        super().__init__(max_vocab_size, max_len, embedding_dim)
        self.model = spacy.load(
            "de_core_news_lg",
            disable=["tagger", "morphologizer", "parser", "lemmatizer", "senter", "attribute_ruler", "ner"],
        )

    def embed(self, X: str) -> Any:
        return self.model(X)[0].vector
