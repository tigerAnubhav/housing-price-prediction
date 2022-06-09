import os
import os.path as op
from distutils.core import setup
from setuptools import PEP420PackageFinder

ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT, "src")


def get_version_info():
    """ Extract version information as a dictionary from version.py. """
    version_info = {}
    # changes made
    version_filename = os.path.join("src", "housing", "version.py")
    with open(version_filename, "r") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
    exec(version_code, version_info)
    return version_info


setup(
    # changes made
    name="housing",
    version=get_version_info()["version"],
    package_dir={"": "src"},
    # changes made
    description="Package for housing price prediction",
    # changes made
    author="Anubhav Yadav",
    packages=PEP420PackageFinder.find(where=str(SRC)),
)
