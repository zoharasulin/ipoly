"""Provide routines for visualising data."""
from typing import Any
from typing import Iterable
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from nptyping import Int
from nptyping import NDArray
from numpy import set_printoptions
from pandas import DataFrame
from sklearn import metrics

# numpy and matplotlib defaults
set_printoptions(threshold=15, linewidth=80)


def plot(
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: Any = None,
    grid: bool = False,
    xticks: Tuple[int] = None,
    yticks: Tuple[int] = None,
):
    """Set the main parameters value."""
    from numpy import arange

    fig = plt.figure(figsize=figsize, dpi=100)
    fig.patch.set_facecolor("xkcd:white")
    plt.xticks(rotation=45, ha="right")

    plt.legend(bbox_to_anchor=(1.05, 1))
    if xticks:
        plt.yticks(arange(*xticks))
    if yticks:
        plt.yticks(arange(*yticks))
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


def plot_history(history, keys: list[str] = None) -> None:
    """Plot the history of a Tensorflow training for each metrics.

    Args:
        history: The history to plot.
        keys: List of all metrics that will be plot.Plot all metrics
            if not specified.
    """
    if keys is None:
        keys = history.history.keys()
    for key in keys:
        plt.plot(history.history[key])
        plt.plot(history.history["val_accuracy"])
        plt.legend(["train", "val"], loc="upper left")
        plot("model accuracy", "epoch", "accuracy")


def plot_correlation_matrix(df: DataFrame) -> None:
    """Plot the correlation matrix of your DataFrame.

    Args:
        df: The input DataFrame.
    """
    from numpy import triu
    from numpy import ones_like

    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = triu(ones_like(corr, dtype=bool))
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
        heatmap.yaxis.get_ticklabels(),
        rotation=0,
        ha="right",
        fontsize=fontsize,
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(),
        rotation=45,
        ha="right",
        fontsize=fontsize,
    )
    axes.set_ylabel("True label")
    axes.set_xlabel("Predicted label")


def plot_confusion_matrix(
    y_pred: NDArray[Any, Int],
    y_true: NDArray[Any, Int],
    labels: Iterable[str] = None,
):
    """Plot the confusion matrix.

    It determines by itself if it has to plot a simple confusion matrix
    or multiple  ones in case of multilabel classification.

    Args:
        y_pred: Predicted logits.
        y_true: True labels.
        labels: Labels name. Need to be specified only if it is a
            multilabel classification.
    """
    from numpy import unique

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    if ((unique(y_true.sum(axis=1)) == 1).all()) or (len(y_true.shape) == 1):
        if len(y_true.shape) == 2:
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
        _confusion_matrix(
            metrics.confusion_matrix(y_true, y_pred),
            ax.flatten(),
            ["N", "Y"],
        )
    else:
        confusion_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
        for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, labels):
            _confusion_matrix(cfs_matrix, axes, ["N", "Y"])
            axes.set_title("Confusion Matrix for the class - " + label)

        fig.tight_layout()
        plt.show()


def batch_to_numpy_images_and_labels(data):
    """Converts TensorFlow batch to numpy arrays for images & labels.

    Args:
        data: A tuple of TensorFlow tensors, where the first tensor represents images and the second tensor represents labels.

    Returns:
        A tuple of two numpy arrays, the first representing images and the second representing labels.
    """
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if (
        numpy_labels.dtype == object
    ):  # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label, classes):
    """Creates a formatted title for a label and its correct label.

    Args:
        label: An integer representing a label.
        correct_label: An integer representing the correct label for comparison.
        classes: A list of class names, indexed by their labels.

    Returns:
        A tuple of a formatted string and a boolean indicating whether the label matches the correct label.
    """
    if correct_label is None:
        return classes[label], True
    correct = label == correct_label
    return (
        "{} [{}{}{}]".format(
            classes[label],
            "OK" if correct else "NO",
            "\u2192" if not correct else "",
            classes[correct_label] if not correct else "",
        ),
        correct,
    )


def display_one_image(
    image,
    title,
    subplot,
    red=False,
    titlesize=16,
) -> Tuple[int, int, int]:
    """Displays a single image with its title.

    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the image.
        subplot (Tuple[int, int, int]): The subplot location to display the image in.
        red (bool, optional): If True, the title text color will be red.
        titlesize (int, optional): The font size of the title text.

    Returns:
        The updated subplot location.
    """
    import matplotlib.pyplot as plt

    plt.subplot(*subplot)
    plt.axis("off")
    plt.imshow(image)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(titlesize) if not red else int(titlesize / 1.2),
            color="red" if red else "black",
            fontdict={"verticalalignment": "center"},
            pad=int(titlesize / 1.5),
        )
    return (subplot[0], subplot[1], subplot[2] + 1)


def display_batch_of_images(databatch, classes, predictions=None):
    """Display batch of images with labels and/or predictions.

    Args:
        databatch: a tuple of (images, labels) or just images
        classes: list of class names
        predictions: list of predicted labels, None if not provided

    Returns:
        Displays images in a grid layout with labels and/or predictions.

    Usage:
        display_batch_of_images(images)
        display_batch_of_images(images, predictions)
        display_batch_of_images((images, labels))
        display_batch_of_images((images,labels), predictions)
    """
    import matplotlib.pyplot as plt
    from math import sqrt

    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    labels = [tf.argmax(label, axis=0) for label in labels]  # One hot to index
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(
        zip(images[: rows * cols], labels[: rows * cols]),
    ):
        title = "" if label is None else classes[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label, classes)
        dynamic_titlesize = (
            FIGSIZE * SPACING / max(rows, cols) * 40 + 3
        )  # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_image(
            image,
            title,
            subplot,
            not correct,
            titlesize=dynamic_titlesize,
        )

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
