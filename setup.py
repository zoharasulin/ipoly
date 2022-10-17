from setuptools import setup, find_packages

setup(
      name='ipoly',
      version='0.0.1',
      license='MIT',
      author="Thomas Danguilhen",
      author_email='thomas.danguilhen@estaca.eu',
      packages=find_packages('ipoly'),
      package_dir={'': 'ipoly'},
      url='https://github.com/Danguilhen/ipoly',
      install_requires=[
            'pyarrow',
            'xlrd',
            'pandas',
            'scipy',
            'typeguard',
            'seaborn',
            'pylatex',
      ],
)
