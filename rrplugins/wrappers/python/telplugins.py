"""
================
telPlugins_C_API
================
Python Wrapper around C API to help avoid use of handle.

Introduction
============
 The plugin API allows users to easily access roadrunner plugins. Only three concepts need to be understood:
 1. Loading a plugin
 2. Setting and Getting plugin properties
 3. Executing the plugin

Before using the plugin system the plugin library must be imported using the line:

>>> import teplugins as tel
To load a plugin called "tel_add_noise" use the Python command:
>>> p = tel.Plugin ("tel_add_noise")
The variable p now represents the plugin. Plugins expose a number of properties, these are variables that can be inspected or set. For
example the add noise plugin has a property called Sigma. To set this to a particular value we would use:
>>> p.Sigma = 0.5
Likewise, properties can be inspected:
>>> p.Sigma
To list all propreties in a plugin use the method listOfProperties. The following uses call together with pprint to make the output
more understandable:
>>> import pprint
>>> pprint.pprint (p.listOfProperties())
[['NoiseType', 'Type of noise (Gaussian = 0, Psychological = 1).'],
['Sigma', 'Standard deviation of applied noise'],
['InputData', 'Data Series to apply the noise to'],
['Progress', 'Indicate progress in (0-100%)']]

To run a plugin so that it carries out its function, use the execute method:
>>> p.execute()
The results from an execute call will either be saved to a file or more likely via properties of the plugin. As a trivial example,
consider a plugin called "add" that has three properties, x, y and z. When the execute() method is called the plugin will
take the values stored in x and y, add them together and store the result in z. The following script illustrates how the plugin would be used from Python:
>>> import teplugins as tel
>>> p = tel.Plugin ("add")
>>> p.x = 1.2
>>> p.y = 4.5
>>> p.execute()
>>> p.z
The following script illustrates a more substantial example using the add_noise plugin

DataSeries
----------
The plugin system supports a special data type called a DataSeries. This is a convenient way to represent rows and colums of
data. The data type also has the ability to label columns with a text string and to associate each value in the data series with an additional
value called a weight. In practice the data series will usually store experimental data and the weights will represent a measure
of undertaintly, perhaps a standard deviation, of the data point. A Data Series can be created using the call:
>>> import teplugins as tel
>>> data = tel.DataSeries()
Data can be entered into a data series either by loading the data from a specially formated file or from a Python NumPy array. For example:
>>> data.readDataSeries ("mydata.txt")
To read numpy arrays into a data series use the code:
>>> import numpy as py
>>> values = py.array([[1,2],{2,5],[3,7]])
>>> data = tel.DataSeries.fromNumPy (values)
The number of rows and columns in a Data Series can be obtained using:
>>> print data.rows
>>> print data.cols
Currently individual values in a data series can be accessed using the set and get methods:
>>> p.setElement (1, 2, 4.567)
>>> print p.getElement (1, 2)
Data series can be plotted using the plot method:
>>> data.plot()
##The following script is an example of using the add_noise plugin. This plugin takes a data series and add a given amount of Guassian noise
to all data except the data in the first column.
>>> import teplugins as *
>>> p = Plugin ("tel_add_noise")
>>> p.viewManual()
>>> pl = p.listOfProperties()
>>> for item in pl:
>>>     print item
>>> p.Sigma = 0.00005
>>> series = DataSeries.loadDataSeries ("..\\Examples\\testData.dat")
>>> series.plot()
>>> p.InputData = series
>>> p.execute()
>>> p.InputData.plot()

Plugins
=======
Plugin objects are instanciated using Plugin class. For example to instanciate a plugin called myplugin, we would
use the code:
>>> p = Plugin ("myplugin")
All interactions with plugins are via plugin properties. Values can be set and retrieved via plugin properties.
For example, if a plugin has a property sigma, we can assign or access the value using the code:
>>> p.sigma = 0.1
>>> print p.sigma
Plugins have a single method that can be used to excecute the plugin's functionality:
>>> p.execute()
Once a plugin has been executed, any output from the plugin can be retrieved via properties. Let's
suppose for example there is a plugin all add, which has three properties called, x, y and result. When executed
the plugin will take the values in x and y, compute the sum and assign it to result. The plugin can therefore
be used as follows:
>>> p = Plugin("add")
>>> p.x = 3.4
>>> p.y = 5.6
>>> p.execute()
>>> p.result

Examples
=========


"""


import rrplugins.telplugins_c_api as tpc
import matplotlib.pyplot as plt
import numpy as np
import os.path
import ctypes

__version__ = "1.2.2"

class DataSeries:
    """DataSeries class for handling roadrunner data types"""
    _data = 0
    _myData = False
    def __init__ (self, handle=None, myData = False):
        """
        Constructor for DataSeries class

        >>> d = DataSeries()
        >>> d = DataSeries (rr)

        Args:
            handle:
            myData:
        """
        if handle == None:
            self._myData = True
            self._data = tpc.createTelluriumData(0, 0)
        else:
            self._data = handle
            self._myData = myData


    @classmethod
    def fromNumPy(cls, numPyData):
        # We have to check the following because roadrunner issues structured arrays
        if numPyData.dtype.names != None:
            # Convert to a unstructured type to obtain array dims
            unstructNumPyData = numPyData.view((float, len (numPyData.dtype.names)))
            nrCols  = unstructNumPyData.shape[1]
            nrRows  = len(unstructNumPyData)
        else:
            nrCols  = numPyData.shape[1]
            nrRows  = len(numPyData)
            unstructNumPyData = numPyData

        if len (unstructNumPyData.shape) < 2:
            raise ValueError ('fromNumPy only accepts two dimensional arrays')

        # If there are no column names then make some up
        if hasattr(numPyData, "colnames") and not numPyData.colnames == None:
            colHdr  = numPyData.colnames
        elif not numPyData.colnames == None:
            colHdr  = numPyData.colnames
        else:
            colHdr = []
            for i in range (nrCols):
                colHdr.append ('x' + str (i))

        columnStr = str(colHdr).strip('[]')
        columnStr = columnStr.strip('()')
        columnStr = columnStr.translate(str.maketrans('','', '\' '))

        dataHandle = tpc.rrpLib.tpCreateTelluriumData(nrRows,nrCols,ctypes.c_char_p(columnStr.encode('utf-8')))

        #Copy the data
        for row in range(nrRows):
            for col in range(nrCols):
                val = unstructNumPyData[row][col]
                tpc.setTelluriumDataElement(dataHandle, row, col, val)
        return cls(dataHandle, True)

    def toNumPy(self):
        return tpc.getNumpyData(self._data)

    def __del__ (self):
        if (self._data != 0):
            try:
                if self._myData == True:
                    tpc.freeTelluriumData (self._data)
                #else:
                #    print 'not freeing data'
            except:
                print("Failed freeing data in DataSeries")
            self._data = 0

    @property
    def __getHandle (self):
        return self._data

    @property
    def rows (self):
        """Use x.rows to get the number of rows"""
        return tpc.getTelluriumDataNumRows(self._data)
    # Use x.toNumpy to get NumPy array

    @property
    def __toNumpy (self):
        return tpc.getNumpyData (self._data)

    @property
    def cols (self):
        """Use x.cols to get the number of columns"""
        return tpc.getTelluriumDataNumCols(self._data)

    def getColumnHeaders (self):
        """
        Retrieve the column headers as a list

        >>> print(d.getColumnHeaders())

        """
        value = tpc.getTelluriumDataColumnHeader(self._data)
        if value == None:
            value = []
        return value

    def getElement (self, row, col):
        """
        Get a specific element from a dataseries
        >>> print(d.getElement (1,2) )
        """
        rowCount = tpc.rrpLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.rrpLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")

        val = ctypes.c_double()
        if tpc.rrpLib.tpGetTelluriumDataElement(self._data, row, col, ctypes.byref(val)) == True:
            return val.value
        else:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # todo Is there a getLastError for this?
            raise Exception("Unable to retrieve element")

    def setElement (self, row, col, value):
        """
        Set a specific element
        >>> d.setElement (1,2, 3.1415)
        """
        rowCount = tpc.rrpLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.rrpLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")
        tpc.setTelluriumDataElement(self._data, row, col, value)

    def getWeight (self, row, col):
        """
        Get the weight for a specific data element
        >>> print d.getWeight (1,2)
        """
        rowCount = tpc.rrpLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.rrpLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")

        if not tpc.hasWeights(self._data):
            raise Exception("This data object do not have any weights allocated. Allocate weights first before using.")

        val = ctypes.c_double()
        if tpc.rrpLib.tpGetTelluriumDataWeight(self._data, row, col, ctypes.byref(val)) == True:
            return val.value
        else:
            # Is there a getLastError for this?
            # TK: When an API function fails, the reason for the failure should always be recorded in a message in getLastError()
            msg = tpc.rrpLib.tpGetLastError()
            raise Exception("Unable to retrieve element. The problem was: {}".format(msg))

    def setWeight (self, row, col, value):
        """
        Set a specific element
        >>> d.setElement (1,2, 3.1415)
        """
        rowCount = tpc.rrpLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.rrpLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")

        if not tpc.hasWeights(self._data):
            raise Exception("This data object do not have any weights allocated. Allocate weights first before using.")

        tpc.setTelluriumDataWeight(self._data, row, col, value)


    @classmethod
    def readDataSeries(cls, fileName):
        """
        Read a dataseries from a file
        >>> d.readDataSeries ("myDataSeries.txt")

        Args:
            fileName:

        Returns:

        """
        if not os.path.isfile (fileName):
            raise Exception ("File not found: " + fileName)
        data = tpc.createTelluriumDataFromFile(ctypes.c_char_p(fileName.encode('utf-8')))
        return cls (data, True)

    def writeDataSeries(self, fileName):
        """
        Write a dataseries to a file

        >>> d.writeDataSeries ("myDataSeries.txt")

        Args:
            fileName:

        Returns:

        """
        tpc.writeTelluriumData(self._data, fileName)

    def plot (self):
        """
        Plot a dataseries as a graph
        >>> d.plot()

        Returns:

        """
        hdr = tpc.getTelluriumDataColumnHeader(self._data)
        npData = tpc.getNumpyData(self._data)
        tpc.plotTelluriumData(npData, hdr)

    def plotBifurcationDiagram (self, bfPoints, bfLabels, legend=True, cmap=None,
                                xlabel=None, ylabel=None, selections=None):
        """
        Plot a dataseries using Bifurcation additional data as input
        >>> d.plot()

        Args:
            bfPoints:
            bfLabels:
            legend:
            cmap:
            xlabel:
            ylabel:
            selections:

        Returns:

        """
        hdr = tpc.getTelluriumDataColumnHeader(self._data)
        npData = tpc.getNumpyData(self._data)
        tpc.plotBifurcationData(npData, hdr, bfPoints, bfLabels, legend, cmap,
                                xlabel, ylabel, selections)


# ------------------------------------------------------------------------

class Event:

    def __init__(self):
        self.handlers = []

    def add(self, handler):
        self.handlers.append(handler)
        return self

    def remove(self, handler):
        self.handlers.remove(handler)
        return self

    def fire(self, sender, earg=None):
        for handler in self.handlers:
            handler(sender, earg)

    __iadd__ = add
    __isub__ = remove
    __call__ = fire



# ------------------------------------------------------------------------

_pluginManager = tpc.createPluginManager()
_pluginsAlreadyLoaded = False

class Plugin:

    _OnProgressList = Event()
    _onProgressEvent = 0
    _propertyNames = []

    def __init__(self, pluginName):
        """
        Create a Plugin instance
        >>> myPlugin = Plugin ("tel_add_noise")

        Args:
            pluginName:
        """
        self.pluginName = pluginName
        self.plugin = tpc.loadPlugin (_pluginManager, pluginName)
        if self.plugin is None:
            raise Exception(tpc.getLastError())
        else:
            lp = self.listOfProperties()
            for element in lp:
                self._propertyNames.append (element[0])
            tpc.resetPlugin(self.plugin)

    def getVersion(self):
        """
        Retrieve the plugins version
        >>> print plugin.getVersion()

        Returns:

        """
        text = tpc.getPluginVersion(self.plugin)
        return text

    def setProperty(self, name, value):
        """
        Set a given propoerty in the plugin.
        >>> myPlugin.setProperty ("Sigma", 0.1)

        Args:
            name:
            value:

        Returns:

        """
        if (isinstance (value, DataSeries)):
            if not tpc.setPluginProperty (self.plugin, name, value._data):
                raise TypeError ("Unable to locate property: ", name)
        else:
            handle  = tpc.getPluginProperty(self.plugin, name);
            if handle == 0:
                raise ValueError ("Unable to locate property: ", name)
            t1 = tpc.getPropertyType (handle)
            if (t1 == "listOfProperties"):
                if isinstance (value, list):
                    if len(value) % 2 != 0:
                        raise TypeError ("Expecting even numbers of elements in the property list")
                    for pair in range(int(len(value)/2)):
                        p = pair*2
                        if not isinstance(value[p], str):
                            raise TypeError("Expecting property name in first element of list")
                        if (not isinstance(value[p+1], float)) and (isinstance(value[p+1], int)):
                            raise TypeError("Expecting floating value in second element of list")
                        para1 = tpc.createProperty(value[p], "double", "", value[p+1])
                        tpc.addPropertyToList (handle, para1)
                else:
                    raise  TypeError ("Expecting a list in setProperty")
            else:
                if not tpc.setPluginProperty (self.plugin, name, value):
                    raise TypeError("Failed to set property {} = {}".format(name, value))

    def getProperty (self, name):
        """
        Get the value for a given propoerty in the plugin.
        >>> print(myPlugin.getProperty("Sigma"))

        Args:
            name:

        Returns:

        """
        handle = tpc.getPluginProperty (self.plugin, name)
        if handle == 0:
            raise ValueError ("Property: " + name + " does not exist")

        value = tpc.getProperty (handle)

        if tpc.getPropertyType(handle) == "telluriumData":
            return DataSeries (value)

        elif tpc.getPropertyType(handle) == "stringList":
            return value.split(',')

        elif tpc.getPropertyType(handle) == "vector<int>":
            val = value.strip('{}')
            val = val.split(',')
            return list(map(int, val))

        elif tpc.getPropertyType(handle) == "matrix":
            dblArray = tpc.getDataArray(value)

            rSize = tpc.getMatrixNumRows(value)
            cSize = tpc.getMatrixNumCols(value)
            length = rSize*cSize
            shape = (rSize, cSize)

            arrPtr = ctypes.cast(dblArray, ctypes.POINTER(ctypes.c_double * length))
            a = np.ctypeslib.as_array((ctypes.c_double * length).from_address(ctypes.addressof(arrPtr.contents)))

            a= a.reshape(rSize,cSize)
            return a

        else:
            return value

    def __setattr__ (self, name, value):
        if (name in self._propertyNames):
            self.setProperty (name, value)
        else:
            super(Plugin, self).__setattr__(name, value)

    def __getattr__ (self, name):
        if name in self._propertyNames:
            return self.getProperty(name)
        else:
            raise AttributeError(name)

    def listOfProperties (self):
        """
        List all the properties in the plugin
        >>> myPlugin.listOfProperties()

        Returns:

        """
        if not self:
            return []
        nameList = tpc.getListOfPluginPropertyNames (self.plugin)
        aList = []
        for i in range (0, len (nameList)):
            name = nameList[i]
            handle = tpc.getPluginProperty(self.plugin, nameList[i])
            hint = tpc.getPropertyHint(handle)
            aList.append ([name, hint])
        return aList

    def listOfPropertyNames(self):
        """
        List all the property names in the plugin
        >>> myPlugin.listOfPropertyNames()

        Returns:

        """
        return tpc.getListOfPluginPropertyNames (self.plugin)

    def listOfPropertyDescriptions (self):
        """
        List all the property descriptions in the plugin
        >>> import pprint
        >>> myPlugin.listOfPropertyDescriptions()
        >>> pprint.pprint (na.listOfProperties())

        Returns:

        """
        nameList = tpc.getListOfPluginPropertyNames (self.plugin)
        aList = []
        for i in range (0, len (nameList)):
            name = nameList[i]
            handle = tpc.getPluginProperty(self.plugin, nameList[i])
            descr = tpc.getPropertyDescription(handle)
            aList.append ([name, descr])
        return aList

    def listOfPropertyHints (self):
        """
        List all the property hints in the plugin
        >>> myPlugin.listOfPropertyHints()

        Returns:

        """
        nameList = tpc.getListOfPluginPropertyNames (self.plugin)
        aList = []
        for i in range (0, len (nameList)):
            name = nameList[i]
            handle = tpc.getPluginProperty(self.plugin, nameList[i])
            descr = tpc.getPropertyHint(handle)
            aList.append ([name, descr])
        return aList


    def loadDataSeriesAsNumPy (self, fileName):
        """
        List all the property hints in the plugin
        >>> myPlugin.listOfPropertyHints()

        Args:
            fileName:

        Returns:

        """
        rrDataHandle = tpc.createTelluriumDataFromFile (fileName)
        return tpc.getNumpyData (rrDataHandle)

    def OnProgress (self, f):
        """
        Make sure garbage collector doens't remove the event pointer

        Args:
            f:

        Returns:

        """
        global _onProgressEvent

        _onProgressEvent =  tpc.NotifyEventEx (f)
        # Pass the address of the self object
        theId = id (self)
        tpc.assignOnProgressEvent(self.plugin, _onProgressEvent, theId, None)


    def execute (self):
        """
        Execute the plugin
        >>> myPlugin.execute()

        Returns:

        """
        if tpc.executePlugin (self.plugin) == True:
            return True
        else:
            raise Exception(tpc.getLastError())

    def executeEx (self, inThread):
        return tpc.executePluginEx (self.plugin, inThread)


    def readAllText(self, fName):
        """
        Read all text from a file
        >>> myplugin.readAllText ("myfile.txt")

        Args:
            fName:

        Returns:

        """
        file = open(fName, 'r')
        str = file.read()
        file.close()
        return str

    def loadPlugins(self):
        tpc.loadPlugins (self.pluginsManager)

    @staticmethod
    def listOfPlugins():
        """
        Static method to list all plugins
        >>> Plugin.listOfPlugins()

        Returns:

        """
        global _pluginsAlreadyLoaded
        # Hack to get round bug in loadPlugins
        if not _pluginsAlreadyLoaded:
            tpc.loadPlugins (_pluginManager)
            _pluginsAlreadyLoaded = True

        aList = []
        names = tpc.getPluginLibraryNames (_pluginManager)
        n = tpc.getNumberOfPlugins (_pluginManager)
        # This is a hack to get round thelack of metadata in the plugin
        # Will be resolved in next revision of library
        for i in range (0, n):
            handle = tpc.getPlugin(_pluginManager, names[i])
            info = tpc.getPluginInfo (handle)
            info = info.split ("\n")
            hint = info[2]
            hint = hint.replace("Category......................", "")
            aList.append ([names[i], hint])
        return aList


    def viewManual (self):
        """
        If a plugin has a manual, view it
        >>> myPlugin.viewManual()

        Returns:

        """
        tpc.displayPluginManual(self.plugin)


    def name (self):
        """
        Returns the name of the plugin
        >>> myPlugin.name()

        Returns:

        """
        return tpc.getPluginName(self.plugin)


    def description (self):
        """
        Returns the description of the plugin
        >>> myPlugin.description()

        Returns:

        """
        return tpc.getPluginDescription(self.plugin)


    def hint (self):
        """
        Returns the hint of the plugin
        >>> myPlugin.hint()
        Returns:

        """
        return tpc.getPluginHint(self.plugin)


    def info (self):
        return tpc.rrpLib.tpGetPluginInfo(self.plugin)

# ----------------------------------------------------------------

def extractColumn (data, index):
    return data[:,index]

def plot (data, myColor="red", myLinestyle="None", myMarker="None", myLabel=""):
    columns = data.shape[1]
    for i in range(columns-1):
        p = plt.plot (data[:,0], data[:,i+1])
        plt.setp (p, color=myColor, marker=myMarker, linestyle = myLinestyle, linewidth=1, label=myLabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        return p

def show():
    plt.show()

def getTelluriumData (rr):
    rrDataHandle = tpc.getTelluriumDataHandle(rr)
    return DataSeries (rrDataHandle)

def getDataSeries (numPyData):
    return DataSeries.fromNumPy(numPyData)

