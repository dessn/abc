from setuptools import setup
from setuptools.command.test import test
import sys

if "test" in sys.argv:
    version = "0.0.0"
else:
    version = "0.0.1"


class PyTest(test):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        test.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    name="Bayesian Supernova Cosmology",
    version=version,
    author="DESSN",
    author_email="samuelreay@gmail.com",
    tests_require=[
        "pytest",
        "pytest-cov"
    ],
    cmdclass = {"test": PyTest},
)