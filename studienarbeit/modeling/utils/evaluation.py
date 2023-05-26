import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def save_metadata(
    class_distribution: dict[str | int, int],
    count_df: dict[str, int] | dict[int, int] | None | None,
    results_folder: str,
):
    """Function to save the metadata of the training to the results folder.

    Saves metadata like the class distribution and the number of samples per party to the results folder.

    Parameters
    ----------
    class_distribution : dict[str  |  int, int]
        The class distribution of the training data.
    count_df : dict[str, int] | dict[int, int] | None, optional
        The number of samples per dataframe (train, test, val). _By default `None`_
    results_folder : str
        The folder to save the results to.
    """
    metadata = {
        "datetime": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "class_distribution": class_distribution,
        "count_df": count_df if count_df is not None else {},
    }

    with open(f"{results_folder}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


def evaluate_test_results(
    y_pred,
    y_test,
    results_folder: str,
    class_distribution: dict[str | int, int],
    count_df: dict[str, int] | dict[int, int] | None = None,
    data_prefix: str = "../..",
):
    """Function to evaluate the results of a classifier and save them to a folder.

    Saves the datetime, additional info, classification report and confusion matrix to the results folder.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted labels.
    y_test : np.ndarray
        The true labels.
    results_folder : str
        The folder to save the results to.
    class_distribution : dict
        The class distribution of the training data.
    count_df : dict[str, int] | dict[int, int] | None, optional
        The number of samples per dataframe (train, test, val). _By default `None`_
    data_prefix : str, optional
        The prefix to the data folder. _By default "../.."_
    """
    with open(f"{data_prefix}/data/party_encoding.json", encoding="utf-8") as f:
        party_encoding = json.load(f)

    y_test_num = np.argmax(y_test, axis=1) if type(y_test) != list and y_test.ndim == 2 else y_test

    Path(results_folder).mkdir(parents=True, exist_ok=True)
    for file in Path(results_folder).glob("*"):
        # Delete all files in the folder to avoid mixing old and new results
        file.unlink()

    save_metadata(class_distribution, count_df, results_folder)

    report = classification_report(y_test_num, y_pred)
    with open(f"{results_folder}/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(str(report))

    conf_mat = confusion_matrix(y_test_num, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=party_encoding.keys())
    disp.plot()
    plt.savefig(f"{results_folder}/confusion_matrix.png", dpi=300)
