from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPooling1D,
    MaxPooling1D,
)
from keras.models import Sequential

from studienarbeit.modeling.approaches.approach import Approach


class CNNApproach(Approach):
    def build_model(self, variation: int) -> Sequential:
        model = Sequential()
        model.add(self.embed_layer)
        match variation:
            case 1:
                model.add(Conv1D(32, 3, padding="same"))
                model.add(Conv1D(32, 3, padding="same"))
                model.add(Conv1D(16, 3, padding="same"))
                model.add(Flatten())
                model.add(Dropout(0.2))
                model.add(Dense(16, activation="relu"))
                model.add(Dropout(0.2))
            case 2:
                model.add(Conv1D(32, 5, padding="same", activation="relu"))
                model.add(MaxPooling1D(5))
                model.add(Conv1D(32, 5, padding="same", activation="relu"))
                model.add(MaxPooling1D(5))
                model.add(Conv1D(32, 5, padding="same", activation="relu"))
                model.add(GlobalMaxPooling1D())
                model.add(Dense(32, activation="relu"))
                model.add(Dropout(0.5))
        model.add(self.output_layer)

        return model
