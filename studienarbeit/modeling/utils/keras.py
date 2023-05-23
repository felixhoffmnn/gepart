import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam


def train_keras_model(model, X_train_vec, y_train, X_val_vec, y_val, X_test_vec, batch_size, epochs, learning_rate):
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    model.summary()
    callbacks = [
        EarlyStopping(monitor="val_loss", verbose=1, patience=3),
    ]

    hist = model.fit(
        X_train_vec,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_val_vec, y_val),
    )

    loss = pd.DataFrame({"train loss": hist.history["loss"], "test loss": hist.history["val_loss"]}).melt()
    loss["epoch"] = loss.groupby("variable").cumcount() + 1
    sns.lineplot(x="epoch", y="value", hue="variable", data=loss).set(title="Model loss", ylabel="")
    plt.show()

    pred = model.predict(X_test_vec)
    y_pred = np.argmax(pred, axis=1)

    return y_pred
