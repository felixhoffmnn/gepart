from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, MaxPooling1D
from keras.models import Sequential

from studienarbeit.modeling.approaches.approach import Approach


class CNNApproach(Approach):
    def build_model(self, variation) -> Sequential:
        model = Sequential()
        model.add(self.embed_layer)

        for _ in range(2):
            model.add(Conv1D(64, 3, padding="same", activation="relu"))
            model.add(MaxPooling1D(5))
            model.add(Dropout(0.3))

        model.add(Conv1D(64, 3, padding="same", activation="relu"))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.3))

        model.add(self.output_layer)

        return model
