#!/usr/bin/env python

import os.path

from setuptools import setup, Distribution

# base directory of file
_basedir = os.path.abspath(os.path.dirname(__file__))

_version_fname = os.path.join(_basedir, 'VERSION.txt')

_version = open(_version_fname).readline().strip()


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(foo):
        return True

# when true, the pip package will have the name libroadrunner[experimental]
# and will be downloadable via pip from the experimental namespace:
#   pip install libroadrunner[experimental]
EXPERIMENTAL = True

setup(
    name='libroadrunner-experimental' if EXPERIMENTAL else "libroadrunner",
    author='J Kyle Medley, Andy Somogyi, Herbert Sauro',
    author_email='tellurium-discuss@u.washington.edu',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: C++',
    ],
    version=_version,
    description='libRoadRunner: A simulation and analysis library for SBML',
    # Read the long description directly from file
    long_description=open("README.md").read().strip(),
    long_description_content_type="text/markdown",
    url='http://libroadrunner.org',
    packages=['roadrunner', 'roadrunner.testing', 'roadrunner.stochastic'],
    package_dir={
        'roadrunner': 'site-packages/roadrunner',
        'roadrunner.testing': 'site-packages/roadrunner/testing',
        'roadrunner.stochastic': 'site-packages/roadrunner/stochastic'
    },
    package_data={
        # add dll, won't hurt unix, not there anyway
        'roadrunner': ['_roadrunner*', '*.so', '*.so.*', '*.dll', '*.txt', '*.dylib', '*.pyd'],
        'roadrunner.testing': ['*.xml', '*.txt', '*.dat', 'dsmts/*.xml', 'dsmts/*.csv', 'test_data/*']
    },
    # read dependencies directly from requirements.txt, Then, manage them
    # only from one place.
    install_requires=open("requirements.txt").read().strip(),
    distclass=BinaryDistribution,
)
