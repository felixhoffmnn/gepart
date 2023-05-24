import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def evaluate_test_results(y_pred, y_test, results_folder, data_prefix: str = "../.."):
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    if (type(y_test) != list) and (y_test.ndim == 2):
        y_test_num = np.argmax(y_test, axis=1)
    else:
        y_test_num = y_test

    report = classification_report(y_test_num, y_pred)
    with open(f"{results_folder}/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(str(report))

    with open(f"{data_prefix}/data/party_encoding.json", encoding="utf-8") as f:
        party_encoding = json.load(f)

    conf_mat = confusion_matrix(y_test_num, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=party_encoding.keys())
    disp.plot()
    plt.savefig(f"{results_folder}/confusion_matrix.png", dpi=300)
