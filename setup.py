from setuptools import setup, find_packages

setup(
    name="datasets",
    version=0.1,
    description="Scripts to load preprocessed datasets (PASCAL3D+, CAVIAR, TownCentre, IDIAP)",
    author="Sergey Prokudin",
    author_email="sergey.prokudin@gmail.com",
    packages=["datasets"],
)

setup(
    name="utils",
    version=0.1,
    description="Misc utils for the project (converters, von Mises losses, etc.)",
    author="Sergey Prokudin",
    author_email="sergey.prokudin@gmail.com",
    packages=["utils"],
)

setup(
    name="models",
    version=0.1,
    description="Keras models for object orientation prediction",
    author="Sergey Prokudin",
    author_email="sergey.prokudin@gmail.com",
    packages=["models"],
)