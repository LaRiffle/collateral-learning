from string import ascii_lowercase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sn


def show_results(results, title, ymin=None, ymax=None):
    plt.figure(figsize=(10, 4))
    vmin, vmax = 10e10, -10e10
    for label, result in results.items():
        plt.plot(range(1, len(result) + 1), result, label=label)
        vmin, vmax = min(vmin, min(result)), max(vmax, max(result))
    plt.ylabel("Accuracy in %")
    plt.xlabel("Epochs")
    plt.ylim(ymin if ymin is not None else vmin, ymax if ymax is not None else vmax)
    plt.title(title)
    plt.legend()
    plt.show()


def build_confusion_matrix(pred_labels, n_labels):
    matrix = np.zeros((n_labels, n_labels))
    for pred, label in pred_labels:
        matrix[label, pred] += 1
    return matrix.astype(int)


def show_confusion_matrix(pred_labels, task="font"):
    if task == "font":
        n_labels = 2
        legend = ["cursive", "Georgia"]
        fig_size = (10, 10)
    else:
        assert task == "char"
        n_labels = 10
        legend = [str(i) for i in range(10)]
        fig_size = (14, 14)

    confusion_matrix = build_confusion_matrix(pred_labels, n_labels)

    conf_mat = pd.DataFrame(confusion_matrix, index=legend, columns=legend)
    plt.figure(figsize=fig_size)
    sn.heatmap(conf_mat, cmap="inferno_r", fmt="d", annot=True)
    plt.savefig("confusion_matrix.png", format="png")
    plt.show()
