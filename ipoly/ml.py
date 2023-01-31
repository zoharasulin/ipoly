"""Provide routines for Machine Learning."""
import random
from typing import Literal
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ipoly.file_management import caster
from ipoly.traceback import raiser


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


def croper(image: np.array, margin: int = 18) -> np.array:
    """Crop the white areason the borders of the input image.

    Args:
        image: The input image.
        margin: The margin in pixels kept around the image.

    Returns:
        The cropped image.
    """
    if len(np.unique(image)) == 1:
        raiser("The image is composed of a single color.")
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
                ),
            )  # complete this line
        return loss

    return weighted_loss


def path(path: str) -> str:
    r"""Replace \ and / of a path according to the user platform."""
    from platform import system as ps

    if ps() == "Windows":
        return path.replace("/", "\\")
    return path.replace("\\", "/")


def interpolator(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate the missing values.

    It does use NearestNDInterpolator from scipy to do the
    interpolation.
    """
    from scipy.interpolate import NearestNDInterpolator

    array = np.array(df)
    filled_array = array[~np.isnan(array).any(axis=1), :]
    for i in range(array.shape[1]):
        idxs = list(range(array.shape[1]))
        idxs.pop(i)
        my_interpolator = NearestNDInterpolator(
            filled_array[:, idxs],
            filled_array[:, i],
            rescale=True,
        )
        array[:, i] = np.apply_along_axis(
            lambda row: my_interpolator(*row[idxs]) if np.isnan(row[i]) else row[i],
            1,
            array,
        )
    return caster(pd.DataFrame(array, columns=df.columns, index=df.index))


def subfinder(mylist: list | pd.Index, pattern: list | pd.Index) -> list:
    """Finds all elements of `mylist` that are present in `pattern`.

    Args:
        mylist : list or pd.Index, the list to search in.
        pattern : list or pd.Index, the list of elements to search for.

    Returns:
        A list of elements of `mylist` that are present in `pattern`
    """
    mylist = list(mylist)
    pattern = list(pattern)
    return list(filter(lambda x: x in pattern, mylist))


def prepare_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    y: str | list[str] = "",
    correlation_threshold: float = 0.9,
    missing_rows_threshold: float = 0.9,
    missing_columns_threshold: float = 0.6,
    categories_ratio_threshold: float = 0.1,
    id_correlation_threshold: float = 0.04,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Prepare the table to feed a ML model.

    Args:
        train_df : The training DataFrame to prepare for Machine Learning.
        test_df : The test DataFrame to prepare for Machine Learning. It is prepared the same way train_df is.
        y : Column(s) name(s) of target variable(s).
        correlation_threshold : The column is removed if its correlation with another is higher than this threshold.
        missing_rows_threshold : The row is removed if the proportion of its non-empty cells is lower than this threshold.
        missing_columns_threshold : The column is removed if its proportion of non-empty cells is lower than this threshold.
        categories_ratio_threshold : The column is removed if its proportion of non unique values is higher than this threshold.
        id_correlation_threshold : The column is removed if all its values are unique the mean correlation with other columns is less than this threshold.
        verbose : Print realised actions if True.
    """
    y = [y] if not y is list else y
    if len(y) != len(subfinder(y, train_df.columns)):
        raiser("y variable is not in df columns", Exception)

    train_df = caster(train_df)
    if test_df is not None:
        test_df = caster(
            test_df,
        )  # TODO Modify caster to be able to process multiple dfs at once.
    category_columns = train_df.select_dtypes(include=["category", object]).columns
    # Drop rows with NaNs in the y columns
    if y != [""]:
        for col in y:
            train_df = train_df[train_df[col].notna()]
    # Drop the categorical columns with too much categories
    droped_category_columns = [
        col
        for col in category_columns
        if train_df[col].nunique() / len(train_df) > categories_ratio_threshold
    ]
    train_df = train_df.drop(
        droped_category_columns,
        axis=1,
    )
    if test_df is not None:
        test_df = test_df.drop(
            droped_category_columns,
            axis=1,
        )
    # Convert categorical data to numerical ones
    train_df = pd.get_dummies(train_df)
    if test_df is not None:
        test_df = pd.get_dummies(test_df)
    # Drop columns with not enough data
    empty_columns = train_df.apply(
        lambda col: (1 - col.count() / len(train_df.index)) < missing_columns_threshold,
        axis=0,
    )
    empty_columns = empty_columns[empty_columns].index
    train_df = train_df.loc[
        :,
        empty_columns,
    ]
    if test_df is not None:
        test_df = test_df.loc[
            :,
            [x for x in empty_columns if x not in y],
        ]
    # Drop rows with not enough data
    train_df = train_df[
        train_df.apply(
            lambda row: (1 - row.count() / len(train_df.columns))
            < missing_rows_threshold,
            axis=1,
        )
    ]
    correlate_couples = []
    corr = train_df.corr().abs()
    for col in corr:
        for index, val in corr[col].items():
            if val > correlation_threshold and (index != col):
                if (index, col) not in correlate_couples:
                    correlate_couples.append((col, index))
        if (train_df[col].nunique() == train_df.shape[0]) and (
            (corr[col].sum() - 1) / corr.shape[0] < id_correlation_threshold
        ):  # TODO improve
            # Drop ids columns (unique values with low correlations with other columns)
            if not col in y:
                train_df = train_df.drop(col, axis=1)
                if test_df is not None:
                    test_df = test_df.drop(col, axis=1)
    for couple in correlate_couples:
        # Drop a column if highly correlated with another one
        if not any(elem in couple for elem in y):
            train_df = train_df.drop(couple[0], axis=1)
            if test_df is not None:
                test_df = test_df.drop(couple[0], axis=1)
    if test_df is not None:
        return (
            interpolator(train_df.drop(y, axis=1)),
            train_df[y],
            interpolator(test_df),
        )
    return interpolator(train_df.drop(y, axis=1)), train_df[y]


def get_optimizer(
    type: Literal["Adam", "SGD", "RMSprop", "Adadelta", "Adamax", "Nadam", "Ftrl"],
    learning_rate,
):
    """Returns the selected Tensorflow's optimizer."""
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


def get_lr_callback(strategy):
    """Return a learning rate callback."""
    LR_START = 0.00001
    LR_MAX = 0.00005 * strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 4
    LR_SUSTAIN_EPOCHS = 4
    LR_EXP_DECAY = 0.8

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
                epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
            ) + LR_MIN
        return lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)


def get_callbacks(strategy, model_name):
    """Get the main callbacks for a Tensorflow model."""
    from datetime import datetime
    from os import makedirs

    scheduler = get_lr_callback(strategy)
    logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
    )
    makedirs("checkpoints", exist_ok=True)
    chk_callback = tf.keras.callbacks.ModelCheckpoint(
        f"checkpoints/{model_name}_best.h5",
        save_weights_only=True,
        monitor="val_f1_score",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    return [
        chk_callback,
        tensorboard_callback,
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
    """Load a model and a tokenizer from the 'transformers' library.

    Args:
        overwrite_output_dir : Overwrite the previous training.
        output_dir : Directory to save the training.
        num_labels : The number of predicted labels.
        model_name_or_path : Name or path of the transformer model.
        cache_dir : The directory for caching the operations.
        model_revision : The revision of the transformer model.
        use_auth_token : Whether to use an authentication token or not.
        tokenizer_name : The name of the tokenizer from HuggingFace.
        config_name : The name of the configuration from HuggingFace.
        config_changes : Config cahnges with a different value than the default value of the configuration.
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
                "Use --overwrite_output_dir to continue regardless.",
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


def train(model, strategy, model_name, train, val, epochs, batch_size):
    """Fit a tensorflow model."""
    return model.fit(
        train[0],
        train[1],
        validation_data=val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(strategy, model_name),
    )


def evaluate(model, train, val, test):
    """Evaluate a model for all datasets."""
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


def find_best_model(X: pd.DataFrame, y: pd.DataFrame, seed: int = 42):
    """Finds the best machine learning model for the given dataset.

    Args:
        X : The input dataset.
        y : The target dataset.
        seed : The seed used for reproducibility, by default 42

    Returns:
        The best model, which can be an instance of a classifier.
    """
    from sklearn.ensemble import (
        BaggingClassifier,
        AdaBoostClassifier,
        ExtraTreesClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import (
        RidgeClassifierCV,
        RidgeClassifier,
        SGDClassifier,
        Perceptron,
        PassiveAggressiveClassifier,
        LogisticRegression,
    )
    from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.semi_supervised import LabelSpreading, LabelPropagation
    from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
    from sklearn.svm import NuSVC, LinearSVC, SVC
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV

    set_seed(seed)

    # split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )

    # initialize the LazyClassifier
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

    # fit the classifier with train data
    models, predictions = clf.fit(X_train, X_test, Y_train, Y_test)

    best_acc = 0
    best_model = None

    # initialize the models to be tested
    models = {
        "LGBMClassifier": LGBMClassifier(),
        "XGBClassifier": XGBClassifier(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "SGDClassifier": SGDClassifier(),
        "RidgeClassifierCV": RidgeClassifierCV(),
        "RidgeClassifier": RidgeClassifier(),
        "ExtraTreeClassifier": ExtraTreeClassifier(),
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "LabelSpreading": LabelSpreading(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "NuSVC": NuSVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LinearSVC": LinearSVC(),
        "ExtraTreesClassifier": ExtraTreesClassifier(),
        "BernoulliNB": BernoulliNB(),
        "BaggingClassifier": BaggingClassifier(),
        "GaussianNB": GaussianNB(),
        "LabelPropagation": LabelPropagation(),
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
        "NearestCentroid": NearestCentroid(),
        "Perceptron": Perceptron(),
        "CalibratedClassifierCV": CalibratedClassifierCV(),
        "SVC": SVC(),
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
        "DummyClassifier": DummyClassifier(),
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
    }

    # sort the models by accuracy
    predictions = {
        k: v
        for k, v in sorted(predictions.items(), key=lambda item: item, reverse=True)
    }

    # take the best 3 models
    best_models = dict(list(predictions.items())[4][1][:3])

    # iterate through the models and parameters
    for model_name, _ in best_models.items():
        model = models[model_name]
        # define the parameters for each model
        params = {key: [value] for key, value in model.get_params().items()}

        optimizer = GridSearchCV(estimator=model, param_grid=params, cv=5)
        optimizer.fit(X_train, Y_train)

        # predict on the test set
        Y_pred = optimizer.predict(X_test)

        # calculate the accuracy
        acc = accuracy_score(Y_test, Y_pred)

        # update the best model if necessary
        if acc > best_acc:
            best_acc = acc
            best_model = model
    best_model = best_model.fit(X, y)
    return best_model


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """Binary form of focal loss.

    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf

    Usage:
        model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    from keras import backend as K

    def binary_focal_loss_fixed(y_true, y_pred):
        """Computes the binary focal loss.

        Args:
            y_true: A tensor of the same shape as `y_pred`
            y_pred:  A tensor resulting from a sigmoid

        Returns:
            A scalar tensor representing the binary focal loss for each element in the batch.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)

        return -K.sum(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0),
        )

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """Softmax version of focal loss.

         m.
    FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
        c=1
    where m = number of classes, c = class and o = observation

    Args:
        alpha: The same as weighing factor in balanced cross entropy.
        gamma: Focusing parameter for modulating factor (1-p).

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    from keras import backend as K

    def categorical_focal_loss_fixed(y_true, y_pred):
        """Computes the categorical focal loss.

        Args:
            y_true: A tensor of the same shape as `y_pred`
            y_pred: A tensor resulting from a softmax

        Returns:
            Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed
