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
    name="angles",
    version=0.1,
    description="Misc utils to work with angles (converters, etc.)",
    author="Sergey Prokudin",
    author_email="sergey.prokudin@gmail.com",
    packages=["angles"],
)