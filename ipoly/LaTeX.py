from typing import Literal, Union

import pandas as pd
from pandas import DataFrame
from pylatex import (
    Document,
    Section,
    Subsection,
    Figure,
    Package,
    NoEscape,
    SubFigure,
    Command,
    Table,
)

import os

available_functions = Literal["section", "subsection", "image", "text", "table"]


def nothing(*_args, **_kwargs):
    pass


class LaTeX:
    """
    Class for creating pdf files easily from Python by writing LaTeX code and compiling it.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.doc = None
        self.chain = []

    def _image(
        self,
        func,
        image_path: Union[str, tuple[str]] = None,
        caption: Union[str, tuple[str]] = None,
        centering: bool = True,
    ):
        if type(image_path) is str:
            image_path = tuple(image_path)
            caption = tuple(caption)
        elif caption is None:
            caption = tuple(None for _ in range(0, len(image_path)))
        image_filenames = tuple(
            os.path.join(os.path.dirname(__file__), "..\\" + path)
            for path in image_path
        )

        def image_func(*args, **kwargs):
            with self.doc.create(Figure(position="H")) as _figure:
                if centering:
                    self.doc.append(Command("centering"))
                for image_filename, image_caption in zip(image_filenames, caption):
                    with self.doc.create(
                        SubFigure(width=NoEscape(rf"{1 / len(caption)}\linewidth"))
                    ) as image:
                        image.add_image(
                            image_filename, width=NoEscape(r"0.95\linewidth")
                        )
                        if image_caption:
                            image.add_caption(image_caption)
            func(*args, *kwargs)

        return image_func

    def _section(self, func, name: str):
        def section_func(*args, **kwargs):
            with self.doc.create(Section(name)):
                func(*args, *kwargs)

        return section_func

    def _subsection(self, func, name: str):
        def section_func(*args, **kwargs):
            with self.doc.create(Subsection(name)):
                func(*args, *kwargs)

        return section_func

    def _text(self, func, text: str):
        if text[-4:] == ".txt":
            with open(text, "r") as file:
                text = file.read()

        def text_func(*args, **kwargs):
            self.doc.append(text)
            func(*args, *kwargs)

        return text_func

    def _table(self, func, table: Union[str, DataFrame], name: str = None):
        if type(table) is str:
            try:
                if table[-4:] == ".csv":
                    table = pd.read_csv(table)
                elif table[-4:] == ".pkl":
                    table = pd.read_pickle(table)
                elif table[-5:] == ".xlsx":
                    # noinspection PyArgumentList
                    table = pd.read_excel(table, index_col=0)

            except PermissionError:
                self._raiser(
                    f"The file '{table}' is already open, please close it before trying again."
                )

        table = table.applymap(
            lambda x: "".join(
                c
                for c in x
                if c
                in ["\t", "\n", "\r"]
                + list(map(chr, range(224, 249)))
                + list(map(chr, range(32, 35)))
                + list(map(chr, range(38, 127)))
            )
            if type(x) == str
            else x
        )
        table = table.applymap(
            lambda x: x.replace("&", "\\&").replace("_", "\\_") if type(x) is str else x
        )

        def table_func(*args, **kwargs):
            with self.doc.create(Table(position="H")) as table_figure:
                if name:
                    table_figure.add_caption(name)
                table_figure.append(Command("centering"))
                table_figure.append(NoEscape(table.style.to_latex()))
            func(*args, *kwargs)

        return table_func

    def _chainer(self):
        self.doc = Document(inputenc="latin1")
        self.doc.packages.append(
            Package(
                "geometry",
                options=["tmargin=1cm", "lmargin=1cm", "rmargin=1cm", "bmargin=2cm"],
            )
        )
        self.doc.packages.append(Package("float"))
        self.doc.packages.append(Package("babel", options=["french"]))
        self.doc.append(r"\listoffigures")
        func_chain = nothing
        for func, args, kwargs in reversed(self.chain):
            func_chain = func(func_chain, *args, *kwargs)
        func_chain(nothing)

    def add(self, func: available_functions, *args, mandatory: bool = True, **kwargs):
        funcs = {
            "section": self._section,
            "subsection": self._subsection,
            "image": self._image,
            "text": self._text,
            "table": self._table,
        }
        if func == "image":
            if type(args[0]) == str:
                paths = tuple(args[0])
            else:
                paths = args[0]
            for path in paths:
                if not os.path.exists(
                    os.path.join(os.path.dirname(__file__), "..\\" + path)
                ):
                    if not mandatory:
                        return None
                    self._raiser(f"The path '{path}' doesn't correspond to an image.")
            if (
                type(args[0]) == tuple
                and len(args) > 1
                and len(args[0]) != len(args[1])
            ):
                if not mandatory:
                    return None
                self._raiser(
                    "The number of images must be equal to the number of captions."
                )
        if func == "table":
            if not os.path.exists(
                os.path.join(os.path.dirname(__file__), "..\\" + args[0])
            ):
                if not mandatory:
                    return None
                self._raiser(
                    f"The provided path '{args[0]}' for the table is incorrect."
                )
            elif args[0][-4:] not in (".csv", ".pkl", "xlsx"):
                if not mandatory:
                    return None
                self._raiser(f"The extension of the file '{args[0]}' is not handled.")
        self.chain.append((funcs[func], args, kwargs))

    def generate_pdf(self, filepath: str = None, compiler="pdfLaTeX"):
        self._chainer()
        self.doc.generate_pdf(filepath, clean_tex=True, compiler=compiler)

    def generate_tex(self, filepath: str = None):
        self._chainer()
        self.doc.generate_tex(filepath)
