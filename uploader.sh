#!/bin/bash
echo "Uploading new version to pypi"
python setup.py sdist
twine upload dist/*