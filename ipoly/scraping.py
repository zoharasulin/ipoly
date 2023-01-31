from abc import ABC, abstractmethod
from typing import List
from ipoly.file_management import load, save
from ipoly.traceback import raiser


class Component(ABC):
    @abstractmethod
    def operation(self) -> str:
        pass


class Leaf(Component):
    def __init__(self, driver, scrap_object, file, backward_moves) -> None:
        self.driver = driver
        self.scrap_object = scrap_object
        self.file = file
        self.backward_moves = backward_moves

    def operation(self, df) -> str:
        df = self.scrap_object(self.driver, df, self.file)
        for backward_move in self.backward_moves:
            backward_move = _value_finder(self.driver, backward_move)
            click_element(self.driver, *backward_move)
        return df, 0


class Composite(Component):
    def __init__(
        self, forward_moves, driver, file, scrap_object, backward_moves
    ) -> None:
        self.forward_moves = forward_moves
        self.driver = driver
        self._children: List[Component] = []
        self.file = file
        self.scrap_object = scrap_object
        self.backward_moves = backward_moves

    def add(self, component: Component) -> None:
        self._children.append(component)

    def operation(self, df) -> str:
        from time import sleep

        sleep(2)
        forward_move = _value_finder(self.driver, self.forward_moves[0])
        elements = self.driver.find_elements(
            "xpath",
            "//"
            + forward_move[0]
            + "[@"
            + forward_move[1]
            + "='"
            + forward_move[2]
            + "']",
        ) + self.driver.find_elements(
            "xpath",
            "//"
            + forward_move[0]
            + "[@"
            + forward_move[1]
            + "='"
            + forward_move[2]
            + " ']",
        )
        if not self._children:
            for _ in elements:
                if len(self.forward_moves) == 1:
                    self.add(
                        Leaf(
                            self.driver,
                            self.scrap_object,
                            self.file,
                            self.backward_moves,
                        )
                    )
                else:
                    self.add(
                        Composite(
                            self.forward_moves[1:],
                            self.driver,
                            self.file,
                            self.scrap_object,
                            self.backward_moves,
                        )
                    )
        elements[len(elements) - len(self._children)].click()
        sleep(3)
        df, exploring = self._children[0].operation(df)
        if not exploring:
            self._children = self._children[1:]
        return df, len(self._children)


def _value_finder(driver, tag):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(driver.page_source, "html.parser")
    args, _ = unknown_attributes_finder((tag[0], {tag[1]: tag[2]}), {}, soup)
    return tag[0], tag[1], args[1][tag[1]]


def move(scrap_object, forward_moves, backward_moves):
    if not all(isinstance(el, list) for el in forward_moves):
        forward_moves = [forward_moves]
    if not all(isinstance(el, list) for el in backward_moves):
        backward_moves = [backward_moves]

    def move_function(driver, df, file):
        tree = Composite(forward_moves, driver, file, scrap_object, backward_moves)
        exploring = True
        while exploring:
            df, exploring = tree.operation(df)
        return df

    return move_function


def _scraping_driver(visible: bool = False, size=(1980, 1080)):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    chrome_options = Options()
    if not visible:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"--window-size={size[0]},{size[1]}")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )


def click_element(driver, categorie, classe, information):
    from selenium.common.exceptions import NoSuchElementException
    from time import sleep

    try:
        element = driver.find_element(
            "xpath", "//" + categorie + "[@" + classe + "='" + information + "']"
        )
    except NoSuchElementException:
        element = driver.find_element(
            "xpath", "//" + categorie + "[@" + classe + "='" + information + " ']"
        )
    element.click()
    sleep(3)


def unknown_attributes_finder(args, kwargs, soup):
    new_args = []
    new_kwargs = {}
    unknown_attr = None
    for arg in args:
        if type(arg) == dict:
            try:
                arg["class_"] = arg.pop("class")
            except KeyError:
                pass
            kwargs.update(arg)
        else:
            new_args.append(arg)
    for kwarg in kwargs.items():
        if type(kwarg[1]) == int:
            if unknown_attr:
                raiser("You can't look for multiple unknown attributes.")
            if kwarg[0] == "class_":
                unknown_attr = ("class", kwarg[1])
            else:
                unknown_attr = kwarg
        else:
            new_kwargs[kwarg[0]] = kwarg[1]
    if unknown_attr:
        search = soup.find_all(new_args, new_kwargs)
        search = [
            elem[unknown_attr[0]] for elem in search if elem.has_attr(unknown_attr[0])
        ]
        attr_values = []
        for elem in search:
            if " ".join(elem) not in attr_values:
                attr_values.append(" ".join(elem))
        if attr_values != []:
            new_args.append(
                {unknown_attr[0]: attr_values[min(unknown_attr[1], len(attr_values))]}
            )
    return new_args, new_kwargs


def _find_object(tag, object_type):
    match object_type:
        case "text":
            return tag.text.strip()
        case "href":
            return tag["href"]


def find_all_object(*args, object_type="text", **kwargs):
    from re import compile as re_compile

    def func(soup):
        nonlocal args, kwargs
        args, kwargs = unknown_attributes_finder(args, kwargs, soup)
        args = [
            arg
            if type(arg) != dict
            else {k: re_compile(v + r" *") for k, v in arg.items()}
            for arg in args
        ]
        return [
            _find_object(tag, object_type) for tag in soup.find_all(*args, **kwargs)
        ]

    return func


def scrap(*actions):
    from bs4 import BeautifulSoup
    from pandas import concat, DataFrame

    def scrap_function(driver, df, file):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        scraped = dict()
        for action in actions:
            scraped[action[0]] = action[1](soup)
        df = concat([df, DataFrame(scraped)], ignore_index=True)
        save(df, file)
        return df

    return scrap_function


def scraper(url, file, scrap_object, visible: bool = False, size=(1980, 1080)):
    from time import sleep
    from pandas import DataFrame

    driver = _scraping_driver(visible, size)
    try:
        df = load(file)
    except FileNotFoundError:
        df = DataFrame(dtype="object")

    driver.get(url)
    sleep(3)
    scrap_object(driver, df, file)
