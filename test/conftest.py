import sys
import os


def pytest_configure():
    sys.path.append(os.path.abspath(os.getcwd()))
