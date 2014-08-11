#!/usr/bin/env python

from distutils.core import setup as _setup
from distutils.command.sdist import sdist as _sdist

import platform
import shutil
import sys
import os.path

# base directory of file
_basedir = os.path.abspath(os.path.dirname(__file__))

print("file: " + __file__)

_version_fname = os.path.join(_basedir, "VERSION.txt")

_version = open(_version_fname).readline().strip()

print("packaging / installing pylibroadrunner version:{0}".format(_version))

def _sharedLibExt():
    import platform
    return "so" if platform.system() != "Windows" else "pyd"

def _checkRequirements():
    import platform
    import sys

    if sys.version_info < (2,6):
        raise SystemExit("Error, libRoadRunner {0} requires Python 2.6 or better, detected Python version {1}"
                        .format(_version, sys.version_info))

    try:
        # check for numpy
        import numpy
    except ImportError:
        raise SystemExit("numpy package not found, libRoadRunner requires numpy 1.2.1 or better")

    # splits the string into an array like: ['1', '2', '1']
    nver = numpy.__version__.split('.')
    nver = [int(x) for x in nver]

    if nver < [1,2,1]:
        raise SystemExit("libRoadRunner requires numpy 1.2.1 or better, detected numpy version {0}"
                         .format(nver))

# get the os string
def _getOSString():
    
    
    s = platform.system().lower()
    unix = False

    if s.startswith('darwin'):
        s='macosx'
        unix = True
    elif s.startswith('linux'):
        s='linux'
        unix = True
    elif s.startswith('win'):
        s='win'
    else:
        raise SystemExit("I don't know how to install on {}, only know OSX(x86_64), Linux(i386,x86_64) and Win(32)"
                         .format(s))

    arch = platform.architecture()
    
    if unix and arch[0].startswith('64'):
        arch = 'x86_64'
    elif unix and arch[0].startswith('32'):
        arch = 'i386'
    else:
        arch = '32' #win 32

    return s + "_" + arch
    
_version = _version + "-" + _getOSString()


def _copyDistFiles():
    """
    copy files from the CMake build into the site packages so they get packages
    up and put in the distribution.
    """

    if platform.system().lower().startswith("win"):
        print("copying windows dlls...")
        
        shutil.copyfile("bin/iconv.dll",      "site-packages/roadrunner/iconv.dll")
        shutil.copyfile("bin/msvcp100.dll",   "site-packages/roadrunner/msvcp100.dll")
        shutil.copyfile("bin/msvcr100.dll",   "site-packages/roadrunner/msvcr100.dll")
        shutil.copyfile("bin/libxml2.DLL",    "site-packages/roadrunner/libxml2.dll")
        shutil.copyfile("bin/zlib1.dll",      "site-packages/roadrunner/zlib1.dll")

    import glob
    for f in glob.glob("*.txt"):
        shutil.copyfile(f, "site-packages/roadrunner/" + f)
    
    
  


# custom sdist command to copy different bits
class _RoadRunnerSDist(_sdist):

    def get_file_list (self):
        """Figure out the list of files to include in the source
        distribution, and put it in 'self.filelist'.  This might involve
        reading the manifest template (and writing the manifest), or just
        reading the manifest, or just using the default file set -- it all
        depend
        """

        print("get_file_list...")

        _sdist.get_file_list(self)

        print("files: {0}".format(self.filelist.files))



def setup(*args):

    args = list(args)

    print("version: " + _version)

    _checkRequirements()

    # may be running in a different dir, temp change dir into 
    # dir of setup file
    curdir = os.path.abspath(os.getcwd())

    print("changing dir to " + _basedir)
    os.chdir(_basedir)


    # copy stuff if we are building
    if len(args) > 0 and args[0].lower().find('dist') >= 0:
        print ("we're building...")
        _copyDistFiles()

    # check if we are running in Mac Spyder, and /Users/andy/local/dist/pylibroadrunner-1.1.1-macosx_x86_64/setup.py
    # called with no args
    if sys.exec_prefix.find("Spyder.app") >= 0 and len(args) == 0:
        args.append("install")
        args.append("--prefix=" + sys.exec_prefix)

        print("installing into Spyder using args: " + str(args))


    _setup(name='pylibroadrunner',
        author='Andy Somogyi, Herbert Sauro',
        author_email='somogyie@indiana.edu',
        version=_version,
        description='libRoadRunner SBML JIT compiler and simulation library',
        url='http://libroadrunner.org',
        packages=['roadrunner', 'roadrunner.testing', 'roadrunner.stochastic'],
        package_dir={
            "roadrunner" : "site-packages/roadrunner",
            "roadrunner.testing" : "site-packages/roadrunner/testing",
            "roadrunner.stochastic" : "site-packages/roadrunner/stochastic"
        },
        package_data={
            # add dll, won't hurt unix, not there anyway
            "roadrunner" : ["_roadrunner." + _sharedLibExt(), "*.dll", "*.txt" ],  
            "roadrunner.testing" : ["*.xml", "*.txt", "*.dat", "dsmts/*.xml", "dsmts/*.csv"]
        },
        cmdclass={'sdist' : _RoadRunnerSDist},
           
        # fake out the command line args
        script_name = 'setup.py',
        script_args = args
    )

    #change dir back
    print("changeing back to " + curdir)
    os.chdir(curdir)

    
    

if __name__ == "__main__":
    args = sys.argv[1:]
    setup(*args)


      
