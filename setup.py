from setuptools import find_packages, setup

setup(
    name="vizclean_ds_app",
    version="0.0.0",
    packages=find_packages(exclude=("tests", ".venv", ".pytest_cache")),
)
