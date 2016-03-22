#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, find_packages
import io
import os
import sys

here = os.path.abspath(os.path.dirname('__file__'))

LONG_DESCRIPTION = open(os.path.join(here,'README.rst')).read()
#LICENSE = open(os.path.join(here,'LICENSE')).read()
INITSCRIPT = open(os.path.join(here,'rios_preprocessor','__init__.py')).read().split('\n')
for line in INITSCRIPT:
    if '__version__' in line:
        v_no = line.split('=')[-1].strip(' ')
        if (v_no[0] == v_no[-1]) and v_no.startswith(("'", '"')):
            VERSION = v_no[1:-1]
        else:
            VERSION = v_no
REQUIREMENTS = open(os.path.join(here,'requirements.txt')).read().split('\n')

setup(
    name='rios_preprocessor',
    version=VERSION,
    packages=['rios_preprocessor'],
    author="Leon Baruah",
    author_email="leon.s.baruah@gmail.com",
    license="Modified BSD License",
    description="Preprocessor for the Natural Capital Project's RIOS software package",
    long_description=LONG_DESCRIPTION,
    url="http://github.com/leonbaruah/rios_preprocessor",
    include_package_data=True,
    install_requires=REQUIREMENTS,
    keywords='RIOS',
    platforms='any',
    classifiers = [
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: GIS'
        ],
)