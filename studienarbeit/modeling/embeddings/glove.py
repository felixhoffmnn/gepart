from typing import Any

import numpy as np

from studienarbeit.modeling.embeddings.embedding import Embedding


class GloVe(Embedding):
    def __init__(self, max_vocab_size: int, max_len: int, embedding_dim: int):
        super().__init__(max_vocab_size, max_len, embedding_dim)
        self.embedding_dict = {}
        with open("../../data/embeddings/glove_german_300.txt", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], "float32")
                self.embedding_dict[word] = vectors

    def embed(self, X: str) -> Any:
        return self.embedding_dict.get(X)
