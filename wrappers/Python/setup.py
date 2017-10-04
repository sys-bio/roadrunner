#!/usr/bin/env python

from setuptools import setup

import platform
import shutil
import sys
import os.path

# base directory of file
_basedir = os.path.abspath(os.path.dirname(__file__))

_version_fname = os.path.join(_basedir, 'VERSION.txt')

_version = open(_version_fname).readline().strip()

# def _copyDistFiles():
#     '''
#     copy files from the CMake build into the site packages so they get packages
#     up and put in the distribution.
#     '''
#
#     if platform.system().lower().startswith('win'):
#         print('copying windows dlls...')
#
#         shutil.copyfile('bin/msvcp140.dll',   'site-packages/roadrunner/msvcp140.dll')
#         shutil.copyfile('bin/vcruntime140.dll',   'site-packages/roadrunner/vcruntime140.dll')
#
#     import glob
#     for f in glob.glob('*.txt'):
#         shutil.copyfile(f, 'site-packages/roadrunner/' + f)


setup(name='libroadrunner',
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
    description='libRoadRunner: A simulation  and analysis library for SBML',
    url='http://libroadrunner.org',
    packages=['roadrunner', 'roadrunner.testing', 'roadrunner.stochastic'],
    package_dir={
        'roadrunner' : 'site-packages/roadrunner',
        'roadrunner.testing' : 'site-packages/roadrunner/testing',
        'roadrunner.stochastic' : 'site-packages/roadrunner/stochastic'
    },
    package_data={
        # add dll, won't hurt unix, not there anyway
        'roadrunner' : ['_roadrunner*', '*.so', '*.so.*', '*.dll', '*.txt', '*.dylib', '*.pyd'],
        'roadrunner.testing' : ['*.xml', '*.txt', '*.dat', 'dsmts/*.xml', 'dsmts/*.csv', 'test_data/*']
    },
    install_requires=['numpy>=1.12', 'matplotlib>=1.5'],
)
