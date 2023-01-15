import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import Iterable, Any, Tuple, Literal
from sklearn import metrics
from nptyping import NDArray, Int
import pandas as pd
from ipoly.file_management import caster
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set the seed fot NumPy, TensorFlow and PyTorch.

    Args:
        seed: The seed value to set.

    """

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
    y_pred: NDArray[Any, Int], y_true: NDArray[Any, Int], labels: Iterable[str] = None
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
    """Compute the loss function of the model.

    Args:
        pos_weights : Positive frequencies of weights.
        neg_weights : Negative frequencies of weights.
        epsilon : To not divide by 0.

    Returns:
        The loss classic function for the lost function.

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
    from scipy.interpolate import NearestNDInterpolator

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
    """Prepare the table to feed a ML model.

    Args:
        df : The DaaFrame to prepare for Machine Learning.
        y : Column(s) name(s) of target variable(s).

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


def get_optimizer(
    type: Literal["Adam", "SGD", "RMSprop", "Adadelta", "Adamax", "Nadam", "Ftrl"],
    learning_rate,
):
    from tensorflow import keras

    match type:
        case "Adam":
            optimizer = keras.optimizers.Adam(learning_rate)
        case "SGD":
            optimizer = keras.optimizers.SGD(learning_rate)
        case "RMSprop":
            optimizer = keras.optimizers.RMSprop(learning_rate)
        case "Adadelta":
            optimizer = keras.optimizers.Adadelta(learning_rate)
        case "Adadelta":
            optimizer = keras.optimizers.Adagrad(learning_rate)
        case "Adadelta":
            optimizer = keras.optimizers.Adamax(learning_rate)
        case "Adadelta":
            optimizer = keras.optimizers.Nadam(learning_rate)
        case "Adadelta":
            optimizer = keras.optimizers.Ftrl(learning_rate)
    return optimizer


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, _epoch, _logs=None):
        self.model.save_pretrained(self.output_dir)


def get_callbacks():
    """Get the main callbacks for a Tensorflow model."""

    import tensorflow_addons as tfa
    from datetime import datetime

    def scheduler(_, lr):
        return lr * tf.math.exp(-0.1)

    scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1
    )
    return [
        tensorboard_callback,
        # tfa.callbacks.TQDMProgressBar(),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1),
        scheduler,
    ]


def compile_model(model, is_regression: bool, *args, **kwargs):
    """Compile a Tensorflow model.

    Args:
        model : The model architecture to compile.
        is_regression : Whenever the prediction task is a classification one or a regression one.
    """

    import tensorflow_addons as tfa

    optimizer = get_optimizer(*args, **kwargs)

    if is_regression:
        loss_fn = tf.keras.losses.MeanSquaredError()
        metrics = [tfa.metrics.F1Score(num_classes=2, average="micro")]
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def load_transformers(
    overwrite_output_dir: bool,
    output_dir: str,
    num_labels: int,
    model_name_or_path,
    cache_dir,
    model_revision,
    use_auth_token: bool,
    tokenizer_name: str = None,
    config_name: str = None,
    config_changes: dict = None,
):
    """Load a transformer, a configuration and a tokenizer from 'transformers' library.

    Args:
        tokenizer_name : The name of the tokenizer from HuggingFace.
        config_name : The name of the configuration from HuggingFace.
    """

    from transformers import AutoConfig
    from transformers import AutoTokenizer
    from transformers import TFAutoModelForSequenceClassification
    from transformers.utils import CONFIG_NAME, TF2_WEIGHTS_NAME
    from os import listdir
    from os.path import isdir

    if not tokenizer_name:
        tokenizer_name = model_name_or_path
    if not config_name:
        config_name = model_name_or_path

    checkpoint = None
    if isdir(output_dir) and len(listdir(output_dir)) > 0 and not overwrite_output_dir:
        if (output_dir / CONFIG_NAME).is_file() and (
            output_dir / TF2_WEIGHTS_NAME
        ).is_file():
            checkpoint = output_dir
        else:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )

    if checkpoint is not None:
        config_path = output_dir
    elif config_name:
        config_path = config_name
    else:
        config_path = model_name_or_path
    if num_labels is not None:
        config = AutoConfig.from_pretrained(
            config_path,
            num_labels=num_labels,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )
    if checkpoint is None:
        model_path = model_name_or_path
    else:
        model_path = checkpoint
    if config_changes:
        for key, value in config_changes.items():
            config.__setattr__(key, value)
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )
    except OSError:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            from_pt=True,
            cache_dir=cache_dir,
            revision=model_revision,
            use_auth_token=True if use_auth_token else None,
        )
    return model, tokenizer


def flatten(l):
    """Flatten a list.

    Args:
        l : The list to flatten.

    """

    return [item for sublist in l for item in sublist]


def compute_position(sentence: str | list, target: int, tokenizer) -> int:
    """Compute the first token's position of the targeted word in a sentence.

    Args:
        sentence : The sentence.
        target : The position of the targeted word in the sentence.
        tokenizer : The tokenizer to compute tokens.

    """

    if type(sentence) == str:
        sentence = sentence.split(" ")
    position = 0
    for i in range(target):
        word = sentence[i]
        if i == target:
            break
        position += len(tokenizer.encode(word)) - 2
    return position + 2


def train(model, train, val, epochs, batch_size):
    return model.fit(
        train[0],
        train[1],
        validation_data=val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(),
    )


def evaluate(model, train, val, test):
    train_loss, train_metrics = model.evaluate(train[0], train[1])
    val_loss, val_metrics = model.evaluate(val[0], val[1])
    test_loss, test_metrics = model.evaluate(test[0], test[1])
    return [
        [train_loss, train_metrics],
        [val_loss, val_metrics],
        [test_loss, test_metrics],
    ]


def get_strategy(mixed_precision: bool = False, xla_accelerate: bool = False):
    """Get the Tensorflow strategy.

    Args:
        mixed_precision : Mixed precision is enabled if True.
        xla_accekerate : XLA acceleration is enabled if True.

    Returns:
        The distribute strategy if you are not using a TPU and the experimental distribute strategy if you are.

    """

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    if mixed_precision:
        if tpu:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
        else:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.set_policy(policy)
    if xla_accelerate:
        tf.config.optimizer.set_jit(True)
    return strategy
