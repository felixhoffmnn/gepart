import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import TextVectorization
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# from keras.preprocessing.text import Tokenizer


def load_dataset(dataset: str, num_samples: int, sentence_level: bool = True, data_prefix: str = "../.."):
    def load_party_programs():
        return pd.read_parquet(
            data_dir_pp / ("party_programs_sentence.parquet" if sentence_level else "party_programs.parquet")
        )[["party", "clean_text", "tokenized_text"]]

    def load_speeches():
        return pd.read_parquet(
            data_dir_speeches / ("speeches_sentence.parquet" if sentence_level else "speeches.parquet")
        )[["party", "clean_text", "tokenized_text"]]

    def load_tweets():
        df_tweets = pd.read_parquet(data_dir_tweets / "prep_tweets_sent_full.parquet")[
            ["party", "clean_text", "filter_text"]
        ]
        df_tweets.rename(columns={"filter_text": "tokenized_text"}, inplace=True)

        return df_tweets

    data_dir_pp = Path(f"{data_prefix}/data/party_programs")
    data_dir_speeches = Path(f"{data_prefix}/data/speeches")
    data_dir_tweets = Path(f"{data_prefix}/data/tweets/dataframes")

    if dataset == "speeches":
        df = load_speeches()
    elif dataset == "party_programs":
        df = load_party_programs()
    elif dataset == "tweets":
        df = load_tweets()
    elif dataset == "all":
        df_pp = load_party_programs()
        df_speeches = load_speeches()
        df_tweets = load_tweets()

        if (df_pp | df_speeches | df_tweets) is None:
            logger.error("No dataset found.")
            sys.exit(1)

        df = pd.concat([df_pp, df_speeches, df_tweets], ignore_index=True)
    else:
        logger.error(f"Dataset {dataset} not found.")
        sys.exit(1)

    if num_samples > 0:
        df = df.sample(num_samples)

    logger.info(f"Loaded dataset ({dataset}) with {len(df)} samples.")

    return df


def get_split_data(
    df, text_col="clean_text", label_col="party", num_classes=6, include_validation=True, transform_to_categorical=True
):
    train, test = train_test_split(df, test_size=0.2, shuffle=True)
    if include_validation:
        train, val = train_test_split(train, test_size=0.2, shuffle=True)
        X_val = np.array(val[text_col])
        y_val = np.array(to_categorical(val[label_col], num_classes))

    X_train = np.array(train[text_col])
    X_test = np.array(test[text_col])

    if transform_to_categorical:
        y_train = np.array(to_categorical(train[label_col], num_classes))
        y_test = np.array(to_categorical(test[label_col], num_classes))
    else:
        y_train = np.array(train[label_col])
        y_test = np.array(test[label_col])

    if include_validation:
        return X_train, X_val, X_test, y_train, y_val, y_test  # type: ignore

    return X_train, X_test, y_train, y_test


def process_bert_data(tokenizer, df):
    processed_data = []

    for _, row in tqdm(df.iterrows()):
        text = row["clean_text"]
        text = " ".join(text.split())

        encodings = tokenizer(text, truncation=True, padding=True, max_length=512)

        label = row["party"]

        encodings["label"] = label
        encodings["text"] = text

        processed_data.append(encodings)

    return pd.DataFrame(processed_data)


def get_bert_data(df, tokenizer, sampling="none"):
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.2)

    if sampling != "none":
        X_train, y_train = get_sampled_data(
            np.array(df_train["clean_text"]), np.array(df_train["party"]), sampling=sampling
        )
        df_train = pd.DataFrame({"clean_text": X_train, "party": y_train})

    train_hg = Dataset(pa.Table.from_pandas(process_bert_data(tokenizer, df_train)))
    val_hg = Dataset(pa.Table.from_pandas(process_bert_data(tokenizer, df_val)))
    test_hg = Dataset(pa.Table.from_pandas(process_bert_data(tokenizer, df_test)))

    return train_hg, val_hg, test_hg


def get_vectorized_input_data(X_train, X_val, X_test, max_vocab_size, max_len):
    # tokenizer = Tokenizer(lower=True, split=" ", num_words=max_vocab_size)
    tokenizer = TextVectorization(
        max_tokens=max_vocab_size, standardize="lower", split="whitespace", output_sequence_length=max_len
    )
    tokenizer.fit_on_texts(X_train)

    X_train_vec = tokenizer.texts_to_sequences(X_train)
    X_val_vec = tokenizer.texts_to_sequences(X_val)
    X_test_vec = tokenizer.texts_to_sequences(X_test)

    seq_lengths = [len(x) for x in X_train_vec]
    max_train_len = int(np.percentile(seq_lengths, 99))

    adapted_max_len = min(max_len, max_train_len)
    logger.info(f"Adapted max sequence length: min(99th percentile, fixed max length): {adapted_max_len}")

    X_train_vec_pad = pad_sequences(X_train_vec, maxlen=adapted_max_len, truncating="post", padding="post")
    X_val_vec_pad = pad_sequences(X_val_vec, maxlen=adapted_max_len, truncating="post", padding="post")
    X_test_vec_pad = pad_sequences(X_test_vec, maxlen=adapted_max_len, truncating="post", padding="post")

    return X_train_vec_pad, X_val_vec_pad, X_test_vec_pad, tokenizer.word_index, adapted_max_len


def get_sampled_data(X_train, y_train, sampling="under"):
    if sampling == "under":
        sampler = RandomUnderSampler()
    elif sampling == "over":
        sampler = RandomOverSampler()
    else:
        raise ValueError("Invalid sampling.")

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        reshaped = True
    else:
        reshaped = False

    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    if reshaped:
        X_train_res = X_train_res.reshape(-1)
        y_train_res = y_train_res.reshape(-1)

    return X_train_res, y_train_res


def cache_fasttext_files(X_train, X_test, y_train, y_test):
    if not os.path.exists("cache/"):
        os.mkdir("cache/")

    with open("cache/train_ft.txt", "w", encoding="utf-8") as f:
        for party, text in zip(y_train, X_train):
            f.write(f"__label__{party} {text}\n")

    with open("cache/test_ft.txt", "w", encoding="utf-8") as f:
        for party, text in zip(y_test, X_test):
            f.write(f"__label__{party} {text}\n")


def get_vectorizer(representation, max_features):
    match (representation):
        case "tf-idf":
            return TfidfVectorizer(
                analyzer="word", max_df=0.3, min_df=10, ngram_range=(1, 2), norm="l2", max_features=max_features
            )
        case "bow":
            return CountVectorizer(lowercase=True, max_features=max_features)
        case _:
            raise ValueError("Invalid representation for DNN approach.")
