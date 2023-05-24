from enum import Enum

import fasttext as ft
import fire
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from studienarbeit.modeling.approaches.cnn import CNNApproach
from studienarbeit.modeling.approaches.dnn import DNNApproach
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

MAX_FEATURES = 100000
MAXLEN = 500

# The DATA_PREFIX is intended to be modified by the user.
DATA_PREFIX = "."


class Dataset(str, Enum):
    SPEECHES = "speeches"
    PARTY_PROGRAMS = "party_programs"
    TWEETS = "tweets"
    ALL = "all"


class Sampling(str, Enum):
    NONE = "none"
    OVER = "over"
    UNDER = "under"


def fasttext(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int,
    epochs: int = 5,
    learning_rate: float = 0.1,
    word_ngrams: int = 2,
):
    """Function enabling the training of a fasttext model.

    Fasttext is a library by Facebook Research that uses bag-of-n-grams to classify text.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use for training. Can be one of "speeches", "party_programs", "tweets", "all".
    num_samples : int
        The number of samples to use for training. If 0, all samples are used.
    sampling : Sampling
        The sampling strategy to use. Can be one of "none", "over", "under".
    epochs : int, optional
        The number of epochs to train the model for. _By default `5`_
    learning_rate : float, optional
        The learning rate to use for training. _By default `0.1`_
    word_ngrams : int, optional
        The number of n-grams to use for training. _By default `2`_
    """
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    X_train, X_test, y_train, y_test = get_split_data(df, "tokenized_text", "party", num_classes, False, False)

    if sampling != Sampling.NONE:
        X_train, y_train = get_sampled_data(X_train, y_train, sampling)
        logger.info(f"Sampled data with {sampling} sampling.")

    cache_fasttext_files(X_train, X_test, y_train, y_test)

    model = ft.train_supervised(
        input="cache/train_ft.txt",
        epoch=epochs,
        lr=learning_rate,
        wordNgrams=word_ngrams,
        loss="softmax",
        dim=300,
        pretrainedVectors="studienarbeit/modeling/cc.de.300.vec",
    )

    df_test = pd.DataFrame({"text": X_test, "party": y_test})
    df_test["prediction"] = df_test["text"].apply(lambda x: int(model.predict(x)[0][0].replace("__label__", "")))

    results_folder = f"./results/fasttext/{dataset}/{sentence_level}_{num_samples}_{sampling}_{num_classes}_{epochs}_{learning_rate}_{word_ngrams}"
    class_distribution = pd.Series(y_train).value_counts().to_dict()
    evaluate_test_results(
        np.array(df_test["prediction"]),
        np.array(df_test["party"]),
        results_folder,
        class_distribution,
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
):
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    X_train, X_test, y_train, y_test = get_split_data(df, "tokenized_text", "party", num_classes, False, False)

    vectorizer = get_vectorizer(representation, MAX_FEATURES)

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    if sampling != Sampling.NONE:
        X_train_vec, y_train = get_sampled_data(X_train_vec, y_train, sampling)
        logger.info(f"Sampled data with {sampling}sampling.")

    match (model):
        case "logistic_regression":
            classifier = LogisticRegression(solver="sag", multi_class="multinomial")
        case "linear_svc":
            classifier = LinearSVC(multi_class="crammer_singer")
        case "sgd":
            classifier = SGDClassifier()

    classifier.fit(X_train_vec, y_train)

    y_pred = classifier.predict(X_test_vec)
    results_folder = (
        f"./results/sklearn/{dataset}/{sentence_level}_{num_samples}_{sampling}_{num_classes}_{representation}_{model}"
    )
    class_distribution = pd.Series(y_train).value_counts().to_dict()
    evaluate_test_results(y_pred, y_test, results_folder, class_distribution, DATA_PREFIX)


def dnn(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int,
    variation: int,
    representation: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
):
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(
        df, "tokenized_text", "party", num_classes, True, True
    )

    vectorizer = get_vectorizer(representation, MAX_FEATURES)

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_val_vec = vectorizer.transform(X_val).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    if sampling != Sampling.NONE:
        X_train_vec, y_train = get_sampled_data(X_train_vec, y_train, sampling)
        logger.info(f"Sampled data with {sampling}sampling.")

    dnn_approach = DNNApproach(num_classes, None, X_train_vec.shape[1])
    model = dnn_approach.build_model(variation)

    y_pred = train_keras_model(
        model, X_train_vec, y_train, X_val_vec, y_val, X_test_vec, batch_size, epochs, learning_rate
    )
    results_folder = f"./results/dnn/{dataset}/{sentence_level}_{num_samples}_{sampling}_{num_classes}_{variation}_{representation}_{epochs}_{learning_rate}_{batch_size}"
    class_distribution = pd.Series(y_train).value_counts().to_dict()
    evaluate_test_results(y_pred, y_test, results_folder, class_distribution, DATA_PREFIX)


def cnn(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int,
    variation: int,
    embeddings: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
):
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(df, "clean_text", "party", num_classes, True, True)
    X_train_vec, X_val_vec, X_test_vec, word_index, adapted_max_len = get_vectorized_input_data(
        X_train, X_val, X_test, max_vocab_size=MAX_FEATURES, max_len=MAXLEN
    )

    if sampling != Sampling.NONE:
        X_train_vec, y_train = get_sampled_data(X_train_vec, y_train, sampling)
        logger.info(f"Sampled data with {sampling}sampling.")

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
    model = cnn_approach.build_model(variation)

    y_pred = train_keras_model(
        model, X_train_vec, y_train, X_val_vec, y_val, X_test_vec, batch_size, epochs, learning_rate
    )
    results_folder = f"./results/cnn/{dataset}/{sentence_level}_{num_samples}_{sampling}_{num_classes}_{variation}_{embeddings}_{epochs}_{learning_rate}_{batch_size}"
    class_distribution = pd.Series(y_train).value_counts().to_dict()
    evaluate_test_results(y_pred, y_test, results_folder, class_distribution, DATA_PREFIX)


def bert(
    dataset: Dataset,
    sentence_level: bool,
    num_samples: int,
    sampling: Sampling,
    num_classes: int,
    model_checkpoint: str = "distilbert-base-german-cased",
    epochs: int = 2,
    learning_rate: float = 7e-5,
    batch_size: int = 4,
):
    df = load_dataset(dataset, num_samples, sentence_level, DATA_PREFIX)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)

    train_hg, val_hg, test_hg = get_bert_data(df, tokenizer, sampling)

    training_args = TrainingArguments(
        output_dir="out",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        use_mps_device=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        save_steps=2000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
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

    trainer.train()

    y_pred = trainer.predict(test_hg)
    y_pred = np.argmax(y_pred.predictions, axis=1)
    y_test = test_hg["label"]
    results_folder = f"./results/bert/{dataset}/{sentence_level}_{num_samples}_{sampling}_{num_classes}_{model_checkpoint}_{epochs}_{learning_rate}_{batch_size}"
    class_distribution = pd.Series(train_hg["label"]).value_counts().to_dict()
    evaluate_test_results(y_pred, y_test, results_folder, class_distribution, DATA_PREFIX)


if __name__ == "__main__":
    fire.Fire(component={"fasttext": fasttext, "sklearn": sklearn, "dnn": dnn, "cnn": cnn, "bert": bert})
