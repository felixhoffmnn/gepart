from keras.layers import Dense, Dropout
from keras.models import Sequential

from studienarbeit.modeling.approaches.approach import Approach


class DNNApproach(Approach):
    def build_model(self, variation: int) -> Sequential:
        model = Sequential()
        match variation:
            case 1:
                model.add(Dense(16, input_shape=(self.max_words,), activation="relu"))
                model.add(Dropout(0.5))
                model.add(Dense(16, activation="relu"))
                model.add(Dropout(0.5))
            case 2:
                model.add(Dense(128, input_shape=(self.max_words,), activation="relu"))
                model.add(Dropout(0.2))
            case 3:
                model.add(Dense(256, input_shape=(self.max_words,), activation="relu"))
                model.add(Dropout(0.3))
                model.add(Dense(200, activation="relu"))
                model.add(Dropout(0.3))
                model.add(Dense(160, activation="relu"))
                model.add(Dropout(0.3))
                model.add(Dense(120, activation="relu"))
                model.add(Dropout(0.3))
                model.add(Dense(80, activation="relu"))
                model.add(Dropout(0.3))
        model.add(self.output_layer)

        return model
