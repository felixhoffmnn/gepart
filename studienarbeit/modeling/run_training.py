import time
from enum import Enum
from pathlib import Path

import fasttext as ft
import fire
import numpy as np
import pandas as pd
import torch
import torch.backends.mps
from loguru import logger
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from studienarbeit.modeling.approaches.cnn import CNNApproach
from studienarbeit.modeling.embeddings.glove import GloVe
from studienarbeit.modeling.embeddings.word2vec import Word2Vec
from studienarbeit.modeling.utils.data import (
    cache_fasttext_files,
    get_bert_data,
    get_sampled_data,
    get_split_data,
    get_vectorized_input_data,
    get_vectorizer,
    load_dataset,
)
from studienarbeit.modeling.utils.evaluation import evaluate_test_results
from studienarbeit.modeling.utils.huggingface import compute_metrics
from studienarbeit.modeling.utils.keras import train_keras_model

MAX_FEATURES = 50000
MAX_LEN = 500
DEFAULT_NUM_CLASSES = 6

# The DATA_PREFIX is intended to be modified by the user.
DATA_PREFIX = "."
BASE_PATH = Path("./studienarbeit/modeling")


class Dataset(str, Enum):
    SPEECHES = "speeches"
    PARTY_PROGRAMS = "party_programs"
    TWEETS = "tweets"
    ALL = "all"


class Sampling(str, Enum):
    NONE = "none"
    OVER = "over"
    UNDER = "under"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def _get_available_device() -> Device:
    if torch.cuda.is_available():
        device = Device.CUDA
    elif torch.backends.mps.is_available():
        device = Device.MPS
    else:
        device = Device.CPU

    logger.info(f"Detected {device.value}.")

    return device


def fasttext(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int = DEFAULT_NUM_CLASSES,
    epochs: int = 20,
    learning_rate: float = 0.1,
    word_ngrams: int = 2,
    test_dataset: Dataset | None = None,
) -> None:
    """Function enabling the training of a fasttext model.

    Fasttext is a library by Facebook Research that uses bag-of-n-grams to classify text.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use for training. Can be one of "speeches", "party_programs", "tweets", "all".
    sentence_level : bool
        Whether to use sentence-level or document-level data. Use 0 for document-level and 1 for sentence-level.
    num_samples : int
        The number of samples to use for training. If 0, all samples are used.
    sampling : Sampling
        The sampling strategy to use. Can be one of "none", "over", "under".
    num_classes : int, optional
        The number of classes to use for training. This is not yet implemented but will be used in the future to
        add a neutral class. _By default `6`_
    epochs : int, optional
        The number of epochs to train the model for. _By default `5`_
    learning_rate : float, optional
        The learning rate to use for training. _By default `0.1`_
    word_ngrams : int, optional
        The number of n-grams to use for training. _By default `2`_
    test_dataset : Dataset, optional
        The dataset to use for testing. If not specified, the training dataset is used. _By default `None`_
    """
    results_folder = (
        BASE_PATH
        / "results"
        / "fasttext"
        / dataset
        / f"{sentence_level}_{num_samples}_{sampling}_{num_classes}_{epochs}_{learning_rate}_{word_ngrams}_{test_dataset}"
    )

    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    if test_dataset is None:
        X_train, X_test, y_train, y_test = get_split_data(df, "tokenized_text", "party", num_classes, False, False)
    else:
        X_train, y_train = np.array(df["tokenized_text"]), np.array(df["party"])
        df_test = load_dataset(test_dataset, 0, sentence_level, DATA_PREFIX)
        X_test, y_test = np.array(df_test["tokenized_text"]), np.array(df_test["party"])
        logger.info(f"Using {test_dataset} as test dataset with {len(X_test)} samples.")

    if sampling != Sampling.NONE:
        X_train, y_train = get_sampled_data(X_train, y_train, sampling)
        logger.success(f"Sampled data with {sampling} sampling.")

    count_entries = X_train.shape[0] + X_test.shape[0]
    logger.info(
        f"Train and test data with a ratio of {X_train.shape[0] / count_entries}/{X_test.shape[0] / count_entries}"
    )

    cache_fasttext_files(X_train, X_test, y_train, y_test)

    start_time = time.time()
    model = ft.train_supervised(
        input="cache/train_ft.txt",
        epoch=epochs,
        lr=learning_rate,
        wordNgrams=word_ngrams,
        loss="softmax",
        dim=300,
        pretrainedVectors="studienarbeit/modeling/cc.de.300.vec",
    )
    end_time = time.time()

    df_test = pd.DataFrame({"text": X_test, "party": y_test})
    df_test["prediction"] = df_test["text"].apply(lambda x: int(model.predict(x)[0][0].replace("__label__", "")))

    class_distribution = pd.Series(y_train).value_counts().to_dict()
    duration = end_time - start_time
    count_df = {
        "train": X_train.shape[0],
        "test": X_test.shape[0],
        "total": count_entries,
        "ratio": {"train": X_train.shape[0] / count_entries, "test": X_test.shape[0] / count_entries},
    }

    evaluate_test_results(
        np.array(df_test["prediction"]),
        np.array(df_test["party"]),
        results_folder,
        class_distribution,
        duration,
        count_df,
        DATA_PREFIX,
    )


def sklearn(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int,
    representation: str,
    model: str,
) -> None:
    """Function enabling the training of a sklearn model.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use for training. Can be one of "speeches", "party_programs", "tweets", "all".
    sentence_level : bool
        Whether to use sentence-level or document-level data. Use 0 for document-level and 1 for sentence-level.
    num_samples : int
        The number of samples to use for training. If 0, all samples are used.
    sampling : Sampling
        The sampling strategy to use. Can be one of "none", "over", "under".
    num_classes : int
        The number of classes to use for training. This is not yet implemented but will be used in the future to
        add a neutral class.
    representation : str
        The representation to use for training. Can be "tfidf" or "bow".
    model : str
        The model to use for training. Can be one of "linear_svc", "multinomial_nb", "mlp".
    """
    results_folder = (
        BASE_PATH
        / "results"
        / "sklearn"
        / dataset
        / f"{sentence_level}_{num_samples}_{sampling}_{num_classes}_{representation}_{model}"
    )
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    X_train, X_test, y_train, y_test = get_split_data(df, "tokenized_text", "party", num_classes, False, False)

    vectorizer = get_vectorizer(representation, MAX_FEATURES)

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    if sampling != Sampling.NONE:
        X_train_vec, y_train = get_sampled_data(X_train_vec, y_train, sampling)
        logger.success(f"Sampled data with {sampling}sampling.")

    count_entries = X_train_vec.shape[0] + X_test_vec.shape[0]
    logger.info(
        f"Train and test data with a ratio of {X_train_vec.shape[0] / count_entries}/{X_test_vec.shape[0] / count_entries}"
    )

    match (model):
        case "linear_svc":
            classifier = LinearSVC(multi_class="crammer_singer", max_iter=2000)
        case "multinomial_nb":
            classifier = MultinomialNB()
        case "mlp":
            classifier = MLPClassifier(
                activation="relu",
                hidden_layer_sizes=(100,),
                learning_rate="invscaling",
                learning_rate_init=7e-4,
                max_iter=50,
                tol=5e-5,
                solver="adam",
                verbose=True,
            )
        case _:
            raise ValueError(f"Model {model} not supported.")

    start_time = time.time()
    classifier.fit(X_train_vec, y_train)
    end_time = time.time()

    y_pred = classifier.predict(X_test_vec)

    class_distribution = pd.Series(y_train).value_counts().to_dict()
    duration = end_time - start_time
    count_df = {
        "train": X_train.shape[0],
        "test": X_test.shape[0],
        "total": count_entries,
        "ratio": {"train": X_train_vec.shape[0] / count_entries, "test": X_test_vec.shape[0] / count_entries},
    }

    evaluate_test_results(y_pred, y_test, results_folder, class_distribution, duration, count_df, DATA_PREFIX)


def cnn(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int,
    embeddings: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> None:
    """Function enabling the training of a cnn model.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use for training. Can be one of "speeches", "party_programs", "tweets", "all".
    sentence_level : bool
        Whether to use sentence-level or document-level data. Use 0 for document-level and 1 for sentence-level.
    num_samples : int
        The number of samples to use for training. If 0, all samples are used.
    sampling : Sampling
        The sampling strategy to use. Can be one of "none", "over", "under".
    num_classes : int
        The number of classes to use for training. This is not yet implemented but will be used in the future to
        add a neutral class.
    embeddings : str
        The embeddings to use for training. Can be "glove" or "word2vec".
    epochs : int
        The number of epochs to train the model for.
    learning_rate : float
        The learning rate to use for training.
    batch_size : int
        The batch size to use for training.
    """
    results_folder = (
        BASE_PATH
        / "results"
        / "cnn"
        / dataset
        / f"{sentence_level}_{num_samples}_{sampling}_{num_classes}_{embeddings}_{epochs}_{learning_rate}_{batch_size}"
    )
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(df, "clean_text", "party", num_classes, True, True)
    X_train_vec, X_val_vec, X_test_vec, word_index, adapted_max_len = get_vectorized_input_data(
        X_train, X_val, X_test, max_vocab_size=MAX_FEATURES, max_len=MAX_LEN
    )

    if sampling != Sampling.NONE:
        X_train_vec, y_train = get_sampled_data(X_train_vec, y_train, sampling)
        logger.success(f"Sampled data with {sampling}sampling.")

    count_entries = X_train_vec.shape[0] + X_test_vec.shape[0] + X_val_vec.shape[0]
    logger.info(
        f"Train and test data with a ratio of {X_train_vec.shape[0] / count_entries}/{X_test_vec.shape[0] / count_entries}/{X_val_vec.shape[0] / count_entries}"
    )

    embedding_obj: GloVe | Word2Vec
    match (embeddings):
        case "glove":
            embedding_obj = GloVe(max_vocab_size=MAX_FEATURES, max_len=adapted_max_len, embedding_dim=300)
        case "word2vec":
            embedding_obj = Word2Vec(max_vocab_size=MAX_FEATURES, max_len=adapted_max_len, embedding_dim=300)
        case _:
            raise ValueError("Invalid embeddings.")

    embedding_obj.build_embedding_matrix(word_index)

    cnn_approach = CNNApproach(num_classes, embedding_obj, MAX_FEATURES)
    model = cnn_approach.build_model(None)

    start_time = time.time()
    y_pred = train_keras_model(
        model, X_train_vec, y_train, X_val_vec, y_val, X_test_vec, batch_size, epochs, learning_rate
    )
    end_time = time.time()

    if (type(y_train) != list) and (y_train.ndim == 2):
        y_train = np.argmax(y_train, axis=1)
    class_distribution = pd.Series(y_train).value_counts().to_dict()
    duration = end_time - start_time
    count_df = {
        "train": X_train.shape[0],
        "test": X_test.shape[0],
        "val": X_val.shape[0],
        "total": count_entries,
        "ratio": {
            "train": X_train_vec.shape[0] / count_entries,
            "test": X_test_vec.shape[0] / count_entries,
            "val": X_val_vec.shape[0] / count_entries,
        },
    }

    evaluate_test_results(y_pred, y_test, results_folder, class_distribution, duration, count_df, DATA_PREFIX)


def bert(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int = DEFAULT_NUM_CLASSES,
    model_checkpoint: str = "distilbert-base-german-cased",
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
) -> None:
    """Function enabling the training of a bert model.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use for training. Can be one of "speeches", "party_programs", "tweets", "all".
    sentence_level : bool
        Whether to use sentence-level or document-level data. Use 0 for document-level and 1 for sentence-level.
    num_samples : int
        The number of samples to use for training. If 0, all samples are used.
    sampling : Sampling
        The sampling strategy to use. Can be one of "none", "over", "under".
    num_classes : int, optional
        The number of classes to use for training. This is not yet implemented but will be used in the future to
        add a neutral class. _By default `6`_
    model_checkpoint : str, optional
        The model checkpoint to use for training. _By default `distilbert-base-german-cased`_
    epochs : int, optional
        The number of epochs to train the model for. _By default `3`_
    learning_rate : float, optional
        The learning rate to use for training. _By default `2e-5`_
    batch_size : int, optional
        The batch size to use for training. _By default `16`_
    """
    device = _get_available_device()
    results_folder = (
        BASE_PATH
        / "results"
        / "bert"
        / dataset
        / f"{sentence_level}_{num_samples}_{sampling}_{num_classes}_{model_checkpoint.replace('/', '-')}_{epochs}_{learning_rate}_{batch_size}"
    )
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)

    train_hg, val_hg, test_hg = get_bert_data(df, tokenizer, sampling)

    training_args = TrainingArguments(
        output_dir=results_folder / "out",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        use_mps_device=(device == Device.MPS),
        load_best_model_at_end=True,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hg,
        eval_dataset=val_hg,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    count_entries = train_hg.shape[0] + test_hg.shape[0] + val_hg.shape[0]
    logger.info(
        f"Train and test data with a ratio of {train_hg.shape[0] / count_entries}/{test_hg.shape[0] / count_entries}/{val_hg.shape[0] / count_entries}"
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    y_pred = trainer.predict(test_hg)
    y_pred = np.argmax(y_pred.predictions, axis=1)
    y_test = test_hg["label"]

    class_distribution = pd.Series(train_hg["label"]).value_counts().to_dict()
    count_df = {
        "train": train_hg.shape[0],
        "test": test_hg.shape[0],
        "val": val_hg.shape[0],
        "total": count_entries,
        "ratio": {
            "train": train_hg.shape[0] / count_entries,
            "test": test_hg.shape[0] / count_entries,
            "val": val_hg.shape[0] / count_entries,
        },
    }
    duration = end_time - start_time

    evaluate_test_results(y_pred, y_test, results_folder, class_distribution, duration, count_df, DATA_PREFIX)


if __name__ == "__main__":
    fire.Fire(component={"fasttext": fasttext, "sklearn": sklearn, "cnn": cnn, "bert": bert})
