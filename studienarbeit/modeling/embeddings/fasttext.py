import pickle
from typing import Any

from studienarbeit.modeling.embeddings.embedding import Embedding


class FastText(Embedding):
    def __init__(self, max_vocab_size: int, max_len: int, embedding_dim: int):
        super().__init__(max_vocab_size, max_len, embedding_dim)
        self.model = pickle.load(open("../../data/embeddings/fasttext_german_300.pkl", "rb"))

    def embed(self, X: str) -> Any:
        try:
            return self.model.get_vector(X)
        except KeyError:
            return None
