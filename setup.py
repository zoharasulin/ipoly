from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open('requirements.txt') as f:
    lines = f.readlines()

setup(
    name="ipoly",
    version="0.1.1",
    license="MIT",
    author="Thomas Danguilhen",
    author_email="thomas.danguilhen@estaca.eu",
    packages=["ipoly"],
    url="https://github.com/Danguilhen/ipoly",
    install_requires=[line[:-1] for line in lines],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
