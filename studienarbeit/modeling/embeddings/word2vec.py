from typing import Any

from gensim.models.keyedvectors import load_word2vec_format

from studienarbeit.modeling.embeddings.embedding import Embedding


class Word2Vec(Embedding):
    def __init__(self, max_vocab_size: int, max_len: int, embedding_dim: int):
        super().__init__(max_vocab_size, max_len, embedding_dim)
        self.model = load_word2vec_format("./data/embeddings/word2vec_german_300.model", binary=True)

    def embed(self, X: str) -> Any:
        try:
            return self.model.get_vector(X)
        except KeyError:
            return None
