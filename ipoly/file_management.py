import re
import os
import glob
from typing import Literal, Tuple
import pyarrow.parquet as pq
import xlrd
import pandas as pd
import numpy as np
import pyarrow as pa
from scipy.interpolate import NearestNDInterpolator
from colorama import Fore, Back, Style


def caster(df: pd.DataFrame):
    string_cols = [col for col, col_type in df.dtypes.items() if col_type == "object"]
    if len(string_cols) > 0:
        mask = df.astype(str).apply(
            lambda x: x.str.match(r"(\d{1,4}[-/\\\. ]\d{1,2}[-/\\\. ]\d{2,4})+.*").any()
        )

        def excel_date(x):
            # Warning : Date in the French format
            x = pd.to_datetime(x, dayfirst=True, errors="ignore")
            x = x.apply(
                lambda y: xlrd.xldate.xldate_as_datetime(y, 0)
                if type(y) in (int, float)
                else y
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
        or pd.Series((col.apply(np.int64) == col)).sum() != df.shape[0]
        else col.apply(np.int64),
        axis=0,
    )
    return df


def load(
    file: str,
    sheet: int = 1,
    skiprows=None,
    on: str = "index",
    classic_data=True,
    recursive=True,
):
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
        pathname = directory + "/**/" + file
    else:
        pathname = directory + "/" + file
    files = glob.glob(pathname, recursive=recursive)
    if len(files) > 1:
        print(
            "There are multiple files with '"
            + file
            + "' name in the directory/subdirectories"
        )  # TODO check multi extension
        raise Exception
    elif len(files) == 0:
        extract = pd.DataFrame()
        print("Warning : The file '" + file + "' wasn't found !")
    else:
        file = files[0].split(file)[0] + file
        if file_format in ("xlsx", "xls", "xlsm"):
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
                    + ", please select a valid sheet."
                )
                raise IndexError
            extract.dropna(how="all", inplace=True)
            if len(
                [
                    True
                    for elem in extract.columns
                    if (type(elem) is str and "Unnamed" in elem)
                ]
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
        elif file_format == "pkl":
            if not os.path.isfile(file):
                print("The specified pickle file doesn't exist !")
            extract = pd.read_pickle(file)
        elif file_format == "csv":
            from detect_delimiter import detect

            with open(file) as myfile:
                firstline = myfile.readline()
                delimiter = detect(firstline, default=";")
                myfile.close()
            extract = pd.read_csv(file, delimiter=delimiter)

            extract.dropna(how="all", inplace=True)
            if len(
                [
                    True
                    for elem in extract.columns
                    if (type(elem) is str and "Unnamed" in elem)
                ]
            ) == len(extract.columns):
                extract, extract.columns = (
                    extract.drop(extract.head(1).index),
                    extract.head(1).values.tolist()[0],
                )
            if classic_data:
                if on != "index":
                    extract.dropna(subset=[on], inplace=True)
                extract = caster(extract)
        elif file_format == "parquet":
            extract = pq.read_pandas(file).to_pandas()
        elif file_format in ["png", "jpg"]:
            from cv2 import imread

            img = imread(file)
            if (img[:, :, 0] == img[:, :, 1]).all() and (
                img[:, :, 0] == img[:, :, 2]
            ).all():
                return img[:, :, 0]
            return img
        elif file_format == "bmp":
            import imageio

            return imageio.v3.imread(file)
        elif file_format == "json":
            import json

            with open(file) as user_file:
                file_contents = user_file.read()
            return json.loads(file_contents)
        elif file_format == "txt":
            with open(file) as f:
                lines = f.readlines()
            return lines
        elif file_format == None:  # Directory
            return [
                load(elem, sheet, skiprows, on, classic_data, recursive)
                for elem in glob.glob(file + "/*")
            ]
        else:
            print("I don't handle this file format yet, came back in a decade.")
            raise Exception
        if classic_data:
            if on == "index":
                extract = extract[~extract.index.duplicated(keep="first")]
            elif (not (on in extract)) and (on != None):
                print(
                    "There is no column name '"
                    + on
                    + "' in the sheet "
                    + str(sheet)
                    + " of the file '"
                    + file
                    + "' so I can't load this file? Try to change the 'on' parameter"
                )
                raise Exception
            else:
                extract.drop_duplicates(subset=on, keep="last", inplace=True)
            extract.replace("Null", np.nan, inplace=True)
    return extract


def save(
    object_to_save: pd.DataFrame | np.ndarray | dict,
    file: str,
    sheet="Data",
    keep_index=True,
):  # FIXME column width when save excel
    file_format = file.split(".")[-1]
    if file_format == "xlsx":
        try:
            writer = pd.ExcelWriter(file)
        except PermissionError:
            print(
                "I can't access the file '" + file + "', the "
                "solution may be to close it and retry."
            )
            raise Exception
        try:
            object_to_save.to_excel(writer, sheet_name=sheet, index=keep_index)
        except IOError:
            object_to_save.to_excel(writer, sheet_name=sheet, index=True)
        # for column in object_to_save:
        #    column_length = max(object_to_save[column].astype(str).map(len).max(), len(column))
        #    col_idx = object_to_save.columns.get_loc(column)
        #    writer.sheets[sheet].set_column(col_idx, col_idx, column_length)
        writer.save()
    elif file_format == "pkl":
        pd.to_pickle(object_to_save, "./" + file)
    elif file_format == "parquet":
        caster(object_to_save)
        table = pa.Table.from_pandas(object_to_save, preserve_index=False)
        pq.write_table(table, file + ".parquet")
    elif file_format == "png":
        from PIL import Image

        im = Image.fromarray(object_to_save)
        im.save(file)
    elif file_format == "json":
        with open(file, "w") as outfile:
            outfile.write(object_to_save)
    else:
        print("I don't handle this file format yet, came back in a decade.")
        raise Exception


def merge(
    df: pd.DataFrame,
    file: str | pd.DataFrame,
    sheet=1,
    on="index",
    skiprows=None,
    how: str = "outer",
    save_file: bool = False,
):
    if not (type(df) is pd.DataFrame):
        print("The df parameter must be a DataFrame.")
        raise Exception
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
                print(
                    "You can't merge your data as there are not column '"
                    + on
                    + "' in your already loaded DataFrame."
                )
                raise Exception
        if df.empty:
            merge = dataBase.copy()
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
    ],
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
    if "LIGHT" in bg:
        bg += "_EX"
    if "LIGHT" in fg:
        fg += "_EX"
    bg = Back.__getattribute__(bg)
    colored_text = f"{Fore.__getattribute__(fg)}{text}{Style.RESET_ALL}"
    return colored_text if bg is None else bg + colored_text
