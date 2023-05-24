from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class Embedding(ABC):
    def __init__(self, max_vocab_size: int, max_len: int, embedding_dim: int):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.corpus: List[List[str]]
        self.embedding_matrix: Any

    def build_embedding_matrix(self, word_index: Any):
        vocab_size = len(word_index) + 1
        self.embedding_matrix = np.zeros((vocab_size, self.embedding_dim))

        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue
            embedding_vector = self.embed(word)
            if embedding_vector is not None:
                self.embedding_matrix[idx] = embedding_vector

    @abstractmethod
    def embed(self, X: str):
        pass
