import re

def listDocMethods(fnames=None):
    """
    Go through a list of file names and extract all of the doc strings.

    Returns a dictionary of class names and method names.
    """

    methods = {}

    regstr = """^\s*\.\.\s*(attribute|method|staticmethod)::\s*(\w+)\.(\w+)"""

    r = re.compile(regstr)

    if not fnames:
        fnames = listRoadRunnerModFiles()

    for fname in fnames:
        with open(fname, "r") as f:
            for line in f.readlines():
                match = r.match(line)
                if match:
                    obj = match.group(2)
                    mthd = match.group(3)
                    if not methods.has_key(obj):
                        methods[obj] = []
                    mthds = methods[obj]
                    mthds.append(mthd)

    return methods

def listRoadRunnerModFiles():
    """
    list all the doc files that make up the roadrunner package docs.
    """

    import os.path as path
    import glob

    dname = path.dirname(path.abspath(__file__))

    moddir = path.join("mod_roadrunner")

    glb = path.join(moddir, "*.rst")

    return glob.glob(glb)

def listRoadRunnerMethods(clsNames):
    """
    list the methods in the roadrunner package.

    :param: methods: a dictionary whose keys will be the names of all the
    roadrunner objects to list methods off of. 
    """

    import roadrunner as rr

    methods = {}

    privRe = re.compile("""_\w+""")
    
    for cls in clsNames:
        mthds = eval("dir(rr." + cls + ")")
        mthds = [x for x in mthds if x not in _rr_excludes and not privRe.match(x)]
        methods[cls] = mthds

    return methods

def compareDocsToRoadRunnerModule(fnames=None):
    """
    compare the methods in the roadrunner module to the documented 
    methods, warn if different.
    """

    docmthds = listDocMethods(fnames)
    rrmthds = listRoadRunnerMethods(docmthds.keys())

    for item in rrmthds.items():
        obj = item[0]
        rrmthd = item[1]
        docmthd = docmthds[obj]

        for mthd in rrmthd:
            if docmthd.count(mthd) == 0:
                print("RoadRunner method " + obj + "." + mthd + " is not documented")


    for item in docmthds.items():
        obj = item[0]
        docmthd = item[1]
        rrmthd = rrmthds[obj]

        for mthd in docmthd:
            if rrmthd.count(mthd) == 0:
                print("Documented method " + obj + "." + mthd + " does not exist in RoadRunner module")


    
# do not include these roadrunner package methods
_rr_excludes = [
    "__class__",
    "__del__", 
    "__delattr__",
    "__dict__", 
    "__doc__",
    "__format__", 
    "__getattr__",
    "__getattribute__", 
    "__hash__",
    "__init__", 
    "__module__",
    "__new__", 
    "__reduce__",
    "__reduce_ex__", 
    "__repr__",
    "__setattr__", 
    "__sizeof__",
    "__str__", 
    "__subclasshook__",
    "__swig_destroy__", 
    "__swig_getmethods__",
    "__swig_setmethods__", 
    "__weakref__"]
                
        
        
        


                    

