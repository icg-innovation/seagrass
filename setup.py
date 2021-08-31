#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

# flake8: noqa

import io
import os
import re
import sys
from shutil import rmtree
from glob import glob

from setuptools import find_packages, setup, Command


def _load_requirements(path_dir, file_name='requirements.txt', comment_char='#'):
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    requirements = []
    for line in lines:
        # filer all comments
        if comment_char in line:
            line = line[:line.index(comment_char)].strip()

        # skip directly installed dependencies
        if line.startswith('http'):
            continue

        if line:  # if requirement is not empty
            requirements.append(line)

    return requirements


def _find_optional_installs(requirements_dir):
    requirements_search = os.path.join(requirements_dir, 'requirements-*.txt')
    requirements_list = glob(requirements_search)

    optional_dict = {}
    for requirements_filepath in requirements_list:
        filename = os.path.basename(requirements_filepath)
        optional_name = re.search("requirements\-(\S*?)\.txt", filename).group(1)

        optional_dict[optional_name] = _load_requirements(requirements_dir, filename)

    return optional_dict

root_path = os.path.abspath(os.path.dirname(__file__))
requirements_dir = os.path.join(root_path, 'requirements')

# Package meta-data.
NAME = 'uop-seagrass'
DESCRIPTION = 'A companion Python module for the UoP SDB seagrass project.'
URL = 'https://github.com/Max-FM/seagrass'
EMAIL = 'max.foxley-marrable@port.ac.uk'
AUTHOR = 'Max Foxley-Marrable, Andrew Lundgren'
REQUIRES_PYTHON = '>=3.6.0, <3.8'
VERSION = '0.0.1'

# What packages are required for this module to be executed?
REQUIRED = _load_requirements(requirements_dir, "requirements.txt")

# What packages are optional?
EXTRAS = _find_optional_installs(requirements_dir)


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for
# that!

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(root_path, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(root_path, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload to PyPI."""

    description = 'Build and publish the package to PyPI.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(root_path, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable)
        )

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


class TestUploadCommand(UploadCommand):
    """Support setup.py upload to TestPyPI."""

    description = 'Build and publish the package to TestPyPI.'
    user_options = []

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(root_path, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable)
        )

        self.status('Uploading the package to TestPyPI via Twine…')
        os.system('twine upload --repository testpypi dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['seagrass'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='None',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
        'test-upload': TestUploadCommand
    },
)
