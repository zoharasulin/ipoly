source deactivate && conda remove -n ipoly --all -y && conda create -n ipoly -y && conda activate ipoly && conda install pip -y && pip install -r requirements-dev.txt && pre-commit install
