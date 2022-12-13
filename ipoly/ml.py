import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import Iterable, Any, Tuple
from sklearn import metrics
from nptyping import NDArray, Int
from scipy.interpolate import NearestNDInterpolator
import pandas as pd
from ipoly.file_management import caster


def set_seed(seed: int = 42) -> None:
    from tensorflow import random as tfr
    from torch.backends import cudnn
    from torch.cuda import manual_seed as cuda_manual_seed
    from torch import manual_seed
    from os import environ

    random.seed(seed)
    np.random.seed(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    tfr.set_seed(seed)
    cuda_manual_seed(seed)
    manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def plot_history(history, keys: list[str] = None) -> None:
    if keys is None:
        keys = history.history.keys()
    for key in keys:
        plt.plot(history.history[key])
        plt.plot(history.history["val_accuracy"])
        plt.legend(["train", "val"], loc="upper left")
        plot("model accuracy", "epoch", "accuracy")


def plot_correlation_matrix(df: DataFrame) -> None:
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )


def _confusion_matrix(confusion_matrix, axes, class_names, fontsize=14):

    df_cm = DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    axes.set_ylabel("True label")
    axes.set_xlabel("Predicted label")


def plot_confusion_matrix(
    y_pred: NDArray[Any, Int], y_true: NDArray[Any, Int], labels: Iterable[str]
):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    if ((np.unique(y_true.sum(axis=1)) == 1).all()) or (len(y_true.shape) == 1):
        if len(y_true.shape) == 2:
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
        _confusion_matrix(
            metrics.confusion_matrix(y_true, y_pred), ax.flatten(), ["N", "Y"]
        )
    else:
        confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
        for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, labels):
            _confusion_matrix(cfs_matrix, axes, ["N", "Y"])
            axes.set_title("Confusion Matrix for the class - " + label)

        fig.tight_layout()
        plt.show()


def croper(image: np.array, margin: int = 18) -> np.array:
    """Crop the white areason the borders of the input image.

    Args:
        image: The input image.
        margin: The margin in pixels kept around the image.

    Returns:
        The cropped image.

    """

    if len(np.unique(image)) == 1:
        raise Exception("The image is composed of a single color.")
    if len(image.shape) == 3:
        image_sum = image.sum(axis=2) % 765
    else:
        image_sum = image == 0
    true_points = np.argwhere(image_sum)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    return image[
        max(0, top_left[0] - margin) : bottom_right[0] + 1 + margin,
        max(0, top_left[1] - margin) : bottom_right[1] + 1 + margin,
    ]


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """This function will calculate the loss function of the model

    Input :
        - pos_weights : positive frequencies wights
        - neg_weights : negative frequencies wights
        - epsilon : to not devide by 0
    Output :
        - loss : the loss classic function for the lost function.
    """

    from keras import backend as K
    from tensorflow import cast, float32

    def weighted_loss(y_true, y_pred):
        y_true, y_pred = cast(y_true, float32), cast(y_pred, float32)
        # initialize loss to zero
        loss = 0.0

        for i in range(len([pos_weights])):
            # for each class, add average weighted loss for that class
            loss += K.mean(
                -(
                    (pos_weights * y_true * K.log(y_pred + epsilon))
                    + (neg_weights * (1 - y_true) * K.log(1 - y_pred + epsilon))
                )
            )  # complete this line
        return loss

    return weighted_loss


def plot(
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: Any = None,
    grid: bool = False,
    xticks: Tuple[int] = None,
    yticks: Tuple[int] = None,
):
    fig = plt.figure(figsize=figsize, dpi=100)
    fig.patch.set_facecolor("xkcd:white")
    plt.xticks(rotation=45, ha="right")

    plt.legend(bbox_to_anchor=(1.05, 1))
    if xticks:
        plt.yticks(np.arange(*xticks))
    if yticks:
        plt.yticks(np.arange(*yticks))
    plt.grid(grid)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if title:
        plt.title(title)
    if ".png" in title:
        plt.savefig(title)
    else:
        plt.show()
    plt.close()


def path(path: str) -> str:
    from platform import system as ps

    if ps() == "Windows":
        return path.replace("/", "\\")
    return path.replace("\\", "/")


def interpolator(df: pd.DataFrame) -> pd.DataFrame:
    array = np.array(df)
    filled_array = array[~np.isnan(array).any(axis=1), :]
    for i in range(array.shape[1]):
        idxs = list(range(array.shape[1]))
        idxs.pop(i)
        my_interpolator = NearestNDInterpolator(
            filled_array[:, idxs], filled_array[:, i], rescale=True
        )
        array[:, i] = np.apply_along_axis(
            lambda row: my_interpolator(*row[idxs]) if np.isnan(row[i]) else row[i],
            1,
            array,
        )
    return caster(pd.DataFrame(array, columns=df.columns, index=df.index))


def prepare_table(
    df: pd.DataFrame,
    y: str | list[str] = "",
    correlation_threshold: float = 0.9,
    missing_rows_threshold: float = 0.9,
    missing_columns_threshold: float = 0.6,
    categories_ratio_threshold: float = 0.1,
    id_correlation_threshold: float = 0.04,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the table to feed a ML model.
    :param df: str, Optional
    :param y:
    :param correlation_threshold:
    :param missing_rows_threshold:
    :param missing_columns_threshold:
    :param categories_ratio_threshold:
    :param id_correlation_threshold:
    :return:
    """

    y = [y] if not y is list else y
    category_columns = df.select_dtypes(include=["category", object]).columns
    # Drop rows with NaNs in tye y columns
    if y != [""]:
        for col in y:
            df = df[df[col].notna()]
    # Drop the categorical columns with too much categories
    df = df.drop(
        [
            col
            for col in category_columns
            if df[col].nunique() / len(df) > categories_ratio_threshold
        ],
        axis=1,
    )
    # Convert categorical data to numerical ones
    df = pd.get_dummies(df)
    # Drop columns with not enough data
    df = df.loc[
        :,
        df.apply(
            lambda col: (1 - col.count() / len(df.index)) < missing_columns_threshold,
            axis=0,
        ),
    ]
    # Drop rows with not enough data
    df = df[
        df.apply(
            lambda row: (1 - row.count() / len(df.columns)) < missing_rows_threshold,
            axis=1,
        )
    ]
    correlate_couples = []
    corr = df.corr().abs()
    for col in corr:
        for index, val in corr[col].items():
            if val > correlation_threshold and (index != col):
                if (index, col) not in correlate_couples:
                    correlate_couples.append((col, index))
        if (df[col].nunique() == df.shape[0]) and (
            (corr[col].sum() - 1) / corr.shape[0] < id_correlation_threshold
        ):  # TODO améliorer
            # Drop ids columns (unique values with low correlations with other columns)
            if not col in y:
                df = df.drop(col, axis=1)
    for couple in correlate_couples:
        # Drop a column if highly correlated with another one
        if not any(elem in couple for elem in y):
            df = df.drop(couple[0], axis=1)
    if verbose:
        print("")
    return interpolator(df.drop(y, axis=1)), df[y]
