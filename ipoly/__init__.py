"""Python package making Data Science tasks easier."""
from typeguard.importhook import install_import_hook

with install_import_hook("ipoly"):
    from ipoly.file_management import *
    from ipoly.LaTeX import *
    from ipoly.communication import *
    from ipoly.logger import *
    from ipoly.traceback import *
    from ipoly.ml import *
    from ipoly.scraping import *
    from ipoly.visualisation import *
