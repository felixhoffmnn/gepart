from abc import ABC, abstractmethod

from keras.layers import Dense, Embedding
from keras.models import Sequential

from studienarbeit.modeling.embeddings.embedding import Embedding as EmbeddingObj


class Approach(ABC):
    def __init__(self, num_classes: int, embed_obj: EmbeddingObj | None, max_words: int):
        if embed_obj is not None:
            self.embed_layer = Embedding(
                embed_obj.embedding_matrix.shape[0],
                embed_obj.embedding_matrix.shape[1],
                weights=[embed_obj.embedding_matrix],
                input_length=embed_obj.max_len,
                trainable=False,
            )
        if max_words is not None:
            self.max_words = max_words
        self.output_layer = Dense(num_classes, activation="softmax")

    @abstractmethod
    def build_model(self, variation: int) -> Sequential:
        pass
