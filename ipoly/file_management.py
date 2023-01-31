"""Provide routines for managing data in files."""
import copy
import glob
import os
import re
import string
from collections import Counter
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xlrd
from colorama import Back
from colorama import Fore
from colorama import Style

from ipoly.traceback import raiser

SILLY_DELIMITERS = frozenset(string.ascii_letters + string.digits + ".")


def caster(df: pd.DataFrame):
    """Cast automatically columns in DataFrames.

    The columns with a data type as object are casted automaticallyas float, int or Datatime according to their pattern.

    Args:
        df: The DataFrame to cast.
    """
    string_cols = [col for col, col_type in df.dtypes.items() if col_type == "object"]
    if len(string_cols) > 0:
        mask = df.astype(str).apply(
            lambda x: x.str.match(
                r"(\d{1,4}[-/\\\. ]\d{1,2}[-/\\\. ]\d{2,4})+.*",
            ).any(),
        )

        def excel_date(x):
            # Warning : Date in the French format
            x = pd.to_datetime(x, dayfirst=True, errors="ignore")
            x = x.apply(
                lambda y: xlrd.xldate.xldate_as_datetime(y, 0)
                if type(y) in (int, float)
                else y,
            )
            return x.astype("datetime64[ns]")

        df.loc[:, mask] = df.loc[:, mask].apply(excel_date)
        del mask
    string_cols = [col for col, col_type in df.dtypes.items() if col_type == "object"]
    for col in string_cols:
        try:
            try:
                df[col] = df[col].str.split(",").str.join(".").values.astype(float)
            except AttributeError:
                df[col] = df[col].values.astype(float)
        except (ValueError, TypeError):
            pass

    df = df.apply(
        lambda col: col
        if not (
            col.dtype in [np.dtype("float64"), np.dtype("float32"), np.dtype("float16")]
        )
        or col.isna().any()
        or pd.Series(col.apply(np.int64) == col).sum() != df.shape[0]
        else col.apply(np.int64),
        axis=0,
    )
    return df


def locate_files(file: str, recursive: bool = True) -> Tuple[List[str], str]:
    """Returns a list of file paths matching the given pattern of file name(s).

    Args:
        file: The name (or pattern) of the file to be located.
        recursive: Whether to search recursively through subdirectories.

    Returns:
        list: A list of file paths matching the given file name and directory.
    """
    split_path = file.split("/")
    if len(split_path) == 1:
        directory = "./"
    else:
        file = split_path[-1]
        directory = "/".join(split_path[:-1])
    del split_path
    file_format = file.split(".")[-1]
    if len(file_format) + 1 >= len(file):
        file_format = None
    if recursive:
        pathname = "**/" + directory + "/**/" + file
    else:
        pathname = directory + "/" + file
    return glob.glob(pathname, recursive=recursive), file_format


def load(
    file: str | Iterable[str],
    sheet: int = 1,
    skiprows=None,
    on: str = "index",
    classic_data: bool = True,
    recursive: bool = True,
    has_title: bool = True,
    has_index: bool = True,
    ordered: bool = False,
):
    """Load files or folders for most used file types.

    Supported file extensions are :
        - csv
        - xlsx
        - xls
        - txt
        - png
        - jpg
        - pkl
        - bmp
        - xlsm
        - json
        - parquet
        - wav
        - yaml
        - tfrec

    Args:
        file : The path to the file or a list of file paths.
            If it is a folder, all supported file types will be
            loaded in a list.
        sheet : Sheet number to extract if the file format is xlsx, xls, or xlsm. Defaults to 1.
        skiprows : Number of rows to skip from the start of the file. Defaults to None.
        on : Column name to use as the index for the dataframe. Defaults to 'index'.
        classic_data : Drop duplicated rows and replace Nulls by NaNs Defaults to True.
        recursive : Whether to search for the file in subdirectories. Defaults to True.
        has_title : Whether the file has a title line. Defaults to True.
        has_index : Whether the file has an index column. Defaults to True.
        ordered : Whether to return the data in the order it was stored. Defaults to False.
    """
    if type(file) != str:
        return [
            load(elem, sheet, skiprows, on, classic_data, recursive) for elem in file
        ]
    files, file_format = locate_files(file, recursive)
    if len(files) > 1 and (file_format != "tfrec"):
        raiser(
            "There are multiple files with '"
            + file
            + "' name in the directory/subdirectories",
        )
    elif len(files) == 0:
        print("Warning : The file '" + file + "' wasn't found !")
        return pd.DataFrame()
    if file_format != "tfrec":
        file = files[0].split(file)[0] + file
    match file_format:
        case "tfrec":
            import tensorflow as tf

            AUTO = tf.data.experimental.AUTOTUNE
            ignore_order = tf.data.Options()
            if not ordered:
                ignore_order.experimental_deterministic = (
                    False  # disable order, increase speed
                )
            dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
            return dataset.with_options(
                ignore_order,
            )  # uses data as soon as it streams in, rather than in its original order
        case "xlsx" | "xls" | "xlsm":
            excel = pd.ExcelFile(file)
            sheets = excel.sheet_names
            try:
                if type(sheet) is int:
                    extract = excel.parse(sheets[sheet - 1], skiprows=skiprows)
                else:
                    extract = excel.parse(sheet, skiprows=skiprows)
            except IndexError:
                print(
                    "There is no sheet number "
                    + str(sheet)
                    + ", please select a valid sheet.",
                )
                raise IndexError
            extract.dropna(how="all", inplace=True)
            if len(
                [
                    True
                    for elem in extract.columns
                    if (type(elem) is str and "Unnamed" in elem)
                ],
            ) == len(extract.columns):
                extract, extract.columns = (
                    extract.drop(extract.head(1).index),
                    extract.head(1).values.tolist()[0],
                )
            if classic_data:
                if on != "index" and on is not None:
                    extract.dropna(subset=[on], inplace=True)
                extract = caster(extract)
            extract.set_index(extract.columns[0])
            extract.drop(extract.columns[0], axis=1, inplace=True)
        case "pkl":
            if not os.path.isfile(file):
                print("The specified pickle file doesn't exist !")
            extract = pd.read_pickle(file)
        case "csv":
            with open(file) as myfile:
                firstline = myfile.readline()
                delimiter = detect(firstline, default=";")
                myfile.close()
            extract = pd.read_csv(file, delimiter=delimiter)
            extract.dropna(how="all", inplace=True)  # Drop empty rows
            if not has_title:
                extract = extract.T.reset_index().T.reset_index(drop=True)
            if len(
                [
                    True
                    for elem in extract.columns
                    if (type(elem) is str and "Unnamed" in elem)
                ],
            ) == len(extract.columns):
                extract, extract.columns = (
                    extract.drop(extract.head(1).index),
                    extract.head(1).values.tolist()[0],
                )
            if classic_data:
                if on != "index":
                    extract.dropna(subset=[on], inplace=True)  # Drop rows without label
                extract = caster(extract)
            if has_index:
                extract = extract.set_index(
                    extract.columns[0],
                )  # Set first column as index
        case "parquet":
            extract = pq.read_pandas(file).to_pandas()
        case "png" | "jpg":
            from cv2 import imread

            img = imread(file)
            if (img[:, :, 0] == img[:, :, 1]).all() and (
                img[:, :, 0] == img[:, :, 2]
            ).all():
                return img[:, :, 0]
            return img
        case "bmp":
            import imageio

            return imageio.v3.imread(file)
        case "json":
            import json

            with open(file) as user_file:
                file_contents = user_file.read()
            return json.loads(file_contents)
        case "txt":
            with open(file) as f:
                lines = f.read().splitlines()
            return lines
        case "wav":
            from librosa import load as librosa_load

            return librosa_load(file, sr=None)
        case "yaml":
            import yaml

            with open(file) as stream:
                return yaml.safe_load(stream)
        case None:  # Directory
            return [
                load(elem, sheet, skiprows, on, classic_data, recursive)
                for elem in glob.glob(file + "/*")
            ]
        case default:
            raiser(
                f"I don't handle this file format yet ({default}), came back in a decade.",
            )
    if classic_data:
        if on == "index":
            extract = extract[~extract.index.duplicated(keep="first")]
        elif (not (on in extract)) and (on != None):
            raiser(
                "There is no column name '"
                + on
                + "' in the sheet "
                + str(sheet)
                + " of the file '"
                + file
                + "' so I can't load this file? Try to change the 'on' parameter",
            )
        else:
            extract.drop_duplicates(subset=on, keep="last", inplace=True)
        extract.replace("Null", np.nan, inplace=True)
    return extract


def save(
    object_to_save: pd.DataFrame | np.ndarray | dict,
    file: str,
    sheet="Data",
    keep_index=True,
):
    """Save different object types to different file types.

    Supported file types are:
        - pkl
        - parquet
        - json
        - xlsx
        - png
        - yaml
        - csv

    Args:
        file: The file name.
        sheet: The sheet name the data is saved if saved in an excel.
        keep_index: Keep the indexes in the saved file.

    Raises:
        Exception: If the file can't be accessed or the file type is
        not supported.
    """
    match file.split(".")[-1]:
        case "xlsx":
            try:
                writer = pd.ExcelWriter(file)
            except PermissionError:
                raiser(
                    "I can't access the file '" + file + "', the "
                    "solution may be to close it and retry.",
                )
            try:
                object_to_save.to_excel(writer, sheet_name=sheet, index=keep_index)
            except OSError:
                object_to_save.to_excel(writer, sheet_name=sheet, index=True)
            # for column in object_to_save:
            #    column_length = max(object_to_save[column].astype(str).map(len).max(), len(column))
            #    col_idx = object_to_save.columns.get_loc(column)
            #    writer.sheets[sheet].set_column(col_idx, col_idx, column_length)
            writer._save()
        case "csv":
            object_to_save.to_csv("./" + file, index=keep_index)
        case "pkl":
            pd.to_pickle(object_to_save, "./" + file)
        case "parquet":
            caster(object_to_save)
            table = pa.Table.from_pandas(object_to_save, preserve_index=False)
            pq.write_table(table, file + ".parquet")
        case "png":
            from PIL import Image

            im = Image.fromarray(object_to_save)
            im.save(file)
        case "json":
            with open(file, "w") as outfile:
                outfile.write(object_to_save)
        case "yaml":
            import yaml

            with open(file, "w") as f:
                yaml.dump(object_to_save, f)
        case default:
            raiser(
                f"I don't handle this file format yet ({default}), come back in a decade.",
            )


def merge(
    df: pd.DataFrame,
    file: str | pd.DataFrame,
    sheet=1,
    on: str = "index",
    skiprows=None,
    how: str = "outer",
    save_file: bool = False,
):
    """Merge two pandas DataFrames either on index or a specific column.

    Args:
        df : The first DataFrame to be merged.
        file : The second DataFrame or file to be merged.
        sheet : The sheet number if `file` is a file name. Defaults to 1.
        on : The column to merge on. Defaults to "index".
        skiprows : The number of rows to skip from the file.
        how : The type of merge to be performed. Defaults to "outer".
        save_file : Whether to save the merged DataFrame to file. Defaults to False.

    Returns:
        pd.DataFrame: The merged DataFrame.

    Raises:
        ValueError: If `df` is not a DataFrame or if the specified `on` column doesn't exist in `file`.
    """
    if not (type(df) is pd.DataFrame):
        raiser("The df parameter must be a DataFrame.")
    if type(file) is pd.DataFrame:
        dataBase = file
    else:
        dataBase = load(
            file,
            sheet=sheet,
            skiprows=skiprows,
            on=on,
        )
    if not dataBase.empty:
        columns = list(dataBase.columns)
        if on != "index":
            try:
                columns.remove(on)
            except ValueError:
                raiser(
                    "You can't merge your data as there are no column '"
                    + on
                    + "' in your already loaded DataFrame.",
                )
        if df.empty:
            merge = dataBase.copy()
        elif on != "index":
            merge = dataBase.merge(df, how=how, on=on)
        else:
            merge = dataBase.merge(df, how=how, left_index=True, right_index=True)
        merge = merge.loc[:, ~merge.columns.duplicated()]
        col = list(merge.columns)
        if on != "index":
            col.remove(on)
        merge.dropna(how="all", subset=col, inplace=True)
        del dataBase
        drop_y = list(filter(lambda v: re.match(".*_y$", v), merge.columns))
        keep_x = list(filter(lambda v: re.match(".*_x$", v), merge.columns))
        keep = [name[:-2] for name in keep_x]
        for col_x, col_y in zip(keep_x, drop_y):
            merge[col_x] = merge[col_x].mask(pd.isnull, merge[col_y], errors="ignore")
        merge.drop(drop_y, axis=1, inplace=True)
        merge.rename(columns=dict(zip(keep_x, keep)), inplace=True)
        merge = merge.sort_index()
        merge.replace("Null", np.nan, inplace=True)
    else:
        merge = df
    if save_file:
        save(merge, file)
    else:
        return merge


def color(
    text: str,
    fg: Literal[
        "BLACK",
        "RED",
        "GREEN",
        "YELLOW",
        "BLUE",
        "MAGENTA",
        "CYAN",
        "WHITE",
        "LIGHTBLACK",
        "LIGHTRED",
        "LIGHTGREEN",
        "LIGHTYELLOW",
        "LIGHTBLUE",
        "LIGHTMAGENTA",
        "LIGHTCYAN",
        "LIGHTWHITE",
    ] = "RED",
    bg: Literal[
        "BLACK",
        "RED",
        "GREEN",
        "YELLOW",
        "BLUE",
        "MAGENTA",
        "CYAN",
        "WHITE",
        "LIGHTBLACK",
        "LIGHTRED",
        "LIGHTGREEN",
        "LIGHTYELLOW",
        "LIGHTBLUE",
        "LIGHTMAGENTA",
        "LIGHTCYAN",
        "LIGHTWHITE",
    ] = None,
) -> str:
    r"""Format the text to print it in color.

    Args:
        text: The input text to format.
        fg: The foreground color.
        bg: The background color.

    Examples:
        >>> "This is a " + color("blue text", "BLUE") + " !"
        'This is a \\x1b[34mblue text\\x1b[0m !'

        >>> print("This is a " + color("blue text", "BLUE") + " !")
        This is a \x1b[34mblue text\x1b[0m !

        >>> "This is a " + color("strange color", "GREEN", "WHITE") + " !"
        'This is a \\x1b[47m\\x1b[32mstrange color\\x1b[0m !'
    """
    if bg and ("LIGHT" in bg):
        bg += "_EX"
    if "LIGHT" in fg:
        fg += "_EX"
    colored_text = f"{Fore.__getattribute__(fg)}{text}{Style.RESET_ALL}"
    return colored_text if bg is None else Back.__getattribute__(bg) + colored_text


def _same(sequence):
    elements = iter(sequence)
    try:
        first = next(elements)
    except StopIteration:
        return True
    for el in elements:
        if el != first:
            return False
    return True


def detect(
    text: str,
    default: Optional[str] = None,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = SILLY_DELIMITERS,
) -> Optional[str]:
    r"""Detects the delimiter used in text formats.

    >>> detect(r"looks|like|the vertical bar\nis|the|delimiter\n")
    '|'

    `detect_delimiter.detect()` looks at the text provided to try to
    find an uncommon delimiter, such as ` for whatever reason.

    >>> detect('looks\x10like\x10something stupid\nis\x10the\x10delimiter')
    '\x10'

    When `detect()` doesn't know, it returns `None`:

    >>> text = "not really any delimiters in here.\nthis is just text.\n"
    >>> detect(text)

    It's possible to provide a default, which will be used in that case:

    >>> detect(text, default=',')
    ','

    By default, it will prevent avoid checking alpha-numeric characters
    and the period/full stop character ("."). This can be adjusted via
    the `blacklist` parameter.

    If you believe that you know the delimiter, it's possible to provide
    a list of possible delimiters to check for via the `whitelist` parameter.
    If you don't provide a value, `[',', ';', ':', '|', '\t']` will be checked.
    """
    if whitelist:
        candidates = whitelist
    else:
        candidates = list(",;:|\t")

    sniffed_candidates = Counter()
    likely_candidates = []

    lines = []
    # todo: support streaming
    text_ = copy.copy(text)
    while len(lines) < 5:
        for line in text_.splitlines():
            lines.append(line)

    for c in candidates:
        fields_for_candidate = []

        for line in lines:
            for char in line:
                if char not in blacklist:
                    sniffed_candidates[char] += 1
            fields = line.split(c)
            n_fields = len(fields)

            # if the delimiter isn't present in the
            # first line, it won't be present in the others
            if n_fields == 1:
                break
            fields_for_candidate.append(n_fields)

        if not fields_for_candidate:
            continue

        if _same(fields_for_candidate):
            likely_candidates.append(c)

    # no delimiter found
    if not likely_candidates:
        if whitelist is None and sniffed_candidates:
            new_whitelist = [
                char for (char, _count) in sniffed_candidates.most_common()
            ]
            return detect(text, whitelist=new_whitelist) or default
        return default

    if default in likely_candidates:
        return default

    return likely_candidates[0]
