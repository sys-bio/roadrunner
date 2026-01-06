#!/usr/bin/env python

from setuptools import setup, Distribution

import platform
import shutil
import sys
import os.path

# base directory of file
_basedir = os.path.abspath(os.path.dirname(__file__))

_version_fname = os.path.join(_basedir, 'VERSION.txt')

_version = open(_version_fname).readline().strip()

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True


setup(name='rrplugins',
    author='Debashish Roy, M Totte Karlsson, J Kyle Medley, Herbert Sauro',
    author_email='tellurium-discuss@u.washington.edu',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: C++',
        ],
    version=_version,
    description='rrplugins: Plugins to extend libroadrunner.',
    url='http://libroadrunner.org',
    packages=['rrplugins'],
    package_dir={
        'rrplugins' : 'site-packages/rrplugins',
#        'rrplugins.testing' : 'site-packages/rrplugins/testing',
    },
    package_data={
        # add dll, won't hurt unix, not there anyway
        'rrplugins' : [
              '*.py',
              '*.txt',
              '*/*.so',
              '*/*.so.*',
              '*/*.dll',
              '*/*.lib',
              '*/*.txt',
              '*/*.dylib'],
#        'rrplugins.testing' : ['*.xml', '*.txt', '*.dat', 'dsmts/*.xml', 'dsmts/*.csv', 'test_data/*']
    },
    install_requires=[
          'numpy>=2.4',
          'matplotlib>=3',
          ],
    distclass=BinaryDistribution,
)
