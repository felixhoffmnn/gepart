from typing import Any

import numpy as np
from bpemb import BPEmb

from studienarbeit.modeling.embeddings.embedding import Embedding


class Bytepair(Embedding):
    def __init__(self, max_vocab_size: int, max_len: int, embedding_dim: int):
        super().__init__(max_vocab_size, max_len, embedding_dim)
        self.model = BPEmb(lang="de")

    def embed(self, X: str) -> Any:
        vec = self.model.embed(X).sum(axis=0)
        l2 = np.atleast_1d(np.linalg.norm(vec, 2))
        l2[l2 == 0] = 1
        return vec / l2
