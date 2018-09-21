import pkgutil
import os, sys
import os.path
import glob
import roadrunner

def getData(resource):
    """
    get the contents of a testing resource file.
    """
    return pkgutil.get_data(__name__, resource).decode("utf8")

def abspath(resource):

    # current dir of package

    d = os.path.dirname(__file__)
    ap = os.path.join(d, resource)
    ap = os.path.normpath(ap)

    if os.path.isfile(ap) and os.access(ap, os.R_OK):
        return ap
    else:
        raise Exception("error could not open the path {} for reading".format(ap))

def dir(pattern='*'):
    d = os.path.dirname(__file__)
    return glob.glob(os.path.join(d, pattern))


def getRoadRunner(resource):
    """
    return a RoadRunner instance loaded with one of the test files.
    """
    data = pkgutil.get_data(__name__, resource)
    r = roadrunner.RoadRunner()
    if sys.version_info[0] < 3:
        r.load(data)
    else:
        r.load(data.decode())
    return r
