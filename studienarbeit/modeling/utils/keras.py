import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


def train_keras_model(model, X_train_vec, y_train, X_val_vec, y_val, X_test_vec, batch_size, epochs, learning_rate):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    model.summary()
    callbacks = [
        EarlyStopping(monitor="val_loss", verbose=1, patience=5),
    ]

    model.fit(
        X_train_vec,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_val_vec, y_val),
    )

    pred = model.predict(X_test_vec)
    y_pred = np.argmax(pred, axis=1)

    return y_pred
