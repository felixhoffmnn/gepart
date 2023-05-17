import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def get_split_data(df, text_col="clean_text", label_col="party", num_classes=6):
    train, test = train_test_split(df, test_size=0.2, shuffle=True)
    train, val = train_test_split(train, test_size=0.2, shuffle=True)

    X_train = np.array(train[text_col])
    X_val = np.array(val[text_col])
    X_test = np.array(test[text_col])
    y_train = np.array(to_categorical(train[label_col], num_classes))
    y_val = np.array(to_categorical(val[label_col], num_classes))
    y_test = np.array(to_categorical(test[label_col], num_classes))

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_vectorized_input_data(X_train, X_val, X_test, max_vocab_size=50000, max_len=500):
    tokenizer = Tokenizer(lower=True, split=" ", num_words=max_vocab_size)
    tokenizer.fit_on_texts(X_train)

    X_train_vec = tokenizer.texts_to_sequences(X_train)
    X_val_vec = tokenizer.texts_to_sequences(X_val)
    X_test_vec = tokenizer.texts_to_sequences(X_test)

    X_train_vec_pad = pad_sequences(X_train_vec, maxlen=max_len, truncating="post", padding="post")
    X_val_vec_pad = pad_sequences(X_val_vec, maxlen=max_len, truncating="post", padding="post")
    X_test_vec_pad = pad_sequences(X_test_vec, maxlen=max_len, truncating="post", padding="post")

    return X_train_vec_pad, X_val_vec_pad, X_test_vec_pad, tokenizer.word_index


def get_sampled_data(X_train, y_train, sampling="under"):
    if sampling == "under":
        sampler = RandomUnderSampler()
    elif sampling == "over":
        sampler = RandomOverSampler()
    else:
        raise ValueError("Invalid sampling.")

    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    return X_train_res, y_train_res
