import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam

from studienarbeit.modeling.approaches.cnn import CNNApproach
from studienarbeit.modeling.approaches.dnn import DNNApproach
from studienarbeit.modeling.approaches.lstm import LSTMApproach
from studienarbeit.modeling.embeddings.bytepair import Bytepair
from studienarbeit.modeling.embeddings.fasttext import FastText
from studienarbeit.modeling.embeddings.glove import GloVe
from studienarbeit.modeling.embeddings.spacy_vectors import SpaCy
from studienarbeit.modeling.embeddings.word2vec import Word2Vec
from studienarbeit.modeling.utils.data import get_split_data, get_vectorized_input_data

parser = ArgumentParser(description="Train a LSTM model.")
parser.add_argument(
    "--dataset", type=str, default="all", help="Dataset to use: 'all', 'speeches', 'party_programs' or 'tweets'."
)
parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to use. Default: 0 (all samples)")
parser.add_argument("--num_classes", type=int, default=6, help="Number of classes to predict.")
parser.add_argument("--approach", type=str, default="LSTM", help="'DNN', 'CNN' or 'LSTM'.")
parser.add_argument("--variation", type=int, default=1, help="Variation of the approach.")
parser.add_argument(
    "--embeddings",
    type=str,
    default="glove",
    help="Embeddings: 'bytepair', 'fasttext', 'glove', 'spacy' or 'word2vec'.",
)
parser.add_argument(
    "--representation",
    type=str,
    default="tf-idf",
    help="For a simple 'DNN' approach, either 'tf-idf' or 'bow' is used for text representation.",
)
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")


args = parser.parse_args()
DATASET = args.dataset
NUM_SAMPLES = args.num_samples
NUM_CLASSES = args.num_classes

APPROACH = args.approach
VARIATON = args.variation
EMBEDDINGS = args.embeddings
REPRESENTATION = args.representation

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr

MAX_FEATURES = 100000
MAXLEN = 500


data_dir_pp = Path("../../data/party_programs")
df_pp = pd.read_parquet(data_dir_pp / "party_programs.parquet")[["party", "clean_text", "tokenized_text"]]

data_dir_speeches = Path("../../data/speeches")
df_speeches = pd.read_parquet(data_dir_speeches / "speeches.parquet")[["party", "clean_text", "tokenized_text"]]

data_dir_tweets = Path("../../data/tweets/dataframes/")
df_tweets = pd.read_parquet(data_dir_tweets / "prep_tweets_sent_full.parquet")[["party", "clean_text", "filter_text"]]
df_tweets.rename(columns={"filter_text": "tokenized_text"}, inplace=True)

if DATASET == "all":
    df_full = pd.concat([df_pp, df_speeches, df_tweets], ignore_index=True)
elif DATASET == "speeches":
    df_full = df_speeches
elif DATASET == "party_programs":
    df_full = df_pp
elif DATASET == "tweets":
    df_full = df_tweets

if NUM_SAMPLES > 0:
    df_full = df_full.sample(NUM_SAMPLES)

logger.info(f"Loaded dataset ({DATASET}) with {len(df_full)} samples.")


if APPROACH == "DNN":
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(df_full, "tokenized_text", "party", NUM_CLASSES)
    if REPRESENTATION == "tf-idf":
        vectorizer = TfidfVectorizer(lowercase=True, max_features=MAX_FEATURES)
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_val_vec = vectorizer.transform(X_val).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()

        print(X_train_vec.shape, X_val_vec.shape, X_test_vec.shape)
    elif REPRESENTATION == "bow":
        vectorizer = CountVectorizer(lowercase=True, max_features=MAX_FEATURES)
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_val_vec = vectorizer.transform(X_val).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()
    else:
        raise ValueError("Invalid representation for DNN approach.")
else:
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(df_full, "clean_text", "party", NUM_CLASSES)
    X_train_vec, X_val_vec, X_test_vec, word_index = get_vectorized_input_data(
        X_train, X_val, X_test, max_vocab_size=MAX_FEATURES, max_len=MAXLEN
    )

logger.info("Vectorized input data.")


if APPROACH == "CNN" or APPROACH == "LSTM":
    embeddings: Bytepair | FastText | GloVe | SpaCy | Word2Vec
    match (EMBEDDINGS):
        case "bytepair":
            embeddings = Bytepair(max_vocab_size=MAX_FEATURES, max_len=MAXLEN, embedding_dim=100)
        case "fasttext":
            embeddings = FastText(max_vocab_size=MAX_FEATURES, max_len=MAXLEN, embedding_dim=300)
        case "glove":
            embeddings = GloVe(max_vocab_size=MAX_FEATURES, max_len=MAXLEN, embedding_dim=300)
        case "spacy":
            embeddings = SpaCy(max_vocab_size=MAX_FEATURES, max_len=MAXLEN, embedding_dim=300)
        case "word2vec":
            embeddings = Word2Vec(max_vocab_size=MAX_FEATURES, max_len=MAXLEN, embedding_dim=300)
        case _:
            raise ValueError("Invalid embeddings.")

    embeddings.build_embedding_matrix(word_index)

    logger.info("Loaded embeddings.")

approach: DNNApproach | CNNApproach | LSTMApproach
match (APPROACH):
    case "DNN":
        approach = DNNApproach(NUM_CLASSES, None, MAX_FEATURES)
    case "CNN":
        approach = CNNApproach(NUM_CLASSES, embeddings, MAX_FEATURES)
    case "LSTM":
        approach = LSTMApproach(NUM_CLASSES, embeddings, MAX_FEATURES)
    case _:
        raise ValueError("Invalid approach.")

model = approach.build_model(VARIATON)

logger.info("Built model.")

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=LR),
    metrics=["accuracy"],
)

model.summary()

# now = datetime.now().strftime("%Y-%m-%d_%H%M")
# if APPROACH == "DNN":
#     model_name_prefix = f"{APPROACH}_{VARIATON}_{REPRESENTATION}"
# else:
#     model_name_prefix = f"{APPROACH}_{VARIATON}_{EMBEDDINGS}"
# mc = ModelCheckpoint(
#     "checkpoints/" + model_name_prefix + f"_{DATASET}_{NUM_SAMPLES}_{now}" + "_{epoch:02d}_{val_loss:.4f}.h5",
#     monitor="val_loss",
#     save_best_only=True,
#     verbose=1,
# )
callbacks = [
    EarlyStopping(monitor="val_loss", verbose=1, patience=3),
    # mc,
]

hist = model.fit(
    X_train_vec, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, validation_data=(X_val_vec, y_val)
)


loss = pd.DataFrame({"train loss": hist.history["loss"], "test loss": hist.history["val_loss"]}).melt()
loss["epoch"] = loss.groupby("variable").cumcount() + 1
plt.figure()
sns.lineplot(x="epoch", y="value", hue="variable", data=loss).set(title="Model loss", ylabel="")
plt.show()


pred = model.predict(X_test_vec)
y_pred = np.argmax(pred, axis=1)
y_test_num = np.argmax(y_test, axis=1)

report = metrics.classification_report(y_test_num, y_pred)
print(report)


with open("../../data/party_encoding.json", encoding="utf-8") as f:
    party_encoding = json.load(f)

conf_mat = confusion_matrix(y_test_num, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=party_encoding.keys())
plt.figure()
disp.plot(cmap=plt.cm.Blues)
plt.show()
