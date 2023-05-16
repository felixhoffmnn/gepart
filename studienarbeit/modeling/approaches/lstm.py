from keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
)
from keras.models import Sequential

from studienarbeit.modeling.approaches.approach import Approach


class LSTMApproach(Approach):
    def build_model(self, variation: int) -> Sequential:
        model = Sequential()
        model.add(self.embed_layer)
        match variation:
            case 1:
                model.add(LSTM(32))
                model.add(Dense(64, activation="relu"))
                model.add(Dropout(0.5))
            case 2:
                model.add(Dropout(0.5))
                model.add(
                    Bidirectional(
                        LSTM(
                            units=32,
                            return_sequences=True,
                            dropout=0.25,
                            recurrent_dropout=0,
                            kernel_initializer="glorot_uniform",
                            recurrent_initializer="glorot_uniform",
                        )
                    )
                )
                model.add(GlobalMaxPooling1D())
                model.add(Dense(32, activation="relu"))
                model.add(Dropout(0.5))
            case 3:
                model.add(Bidirectional(LSTM(32, dropout=0.5)))
                model.add(Dropout(0.5))
                model.add(Dense(128, activation="relu"))
            case 4:
                model.add(Bidirectional(LSTM(16, dropout=0.2, return_sequences=True)))
                model.add(GlobalMaxPooling1D())
                model.add(BatchNormalization())
                model.add(Dropout(0.5))
                model.add(Dense(32, activation="relu"))
                model.add(Dropout(0.5))
                model.add(Dense(32, activation="relu"))
                model.add(Dropout(0.5))
        model.add(self.output_layer)

        return model
