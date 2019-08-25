##@Module telPlugins_C_API

## Wrapper around C API to help avoid use of handle.
# An example is given at the end of the code.

import rrplugins.telplugins_c_api as tpc
import matplotlib.pyplot as plt
import numpy as np
import os.path
import ctypes

__version__ = "1.2.2"

## \brief DataSeries class for handling roadrunner data types
class DataSeries(object):

    _data = 0
    _myData = False
    ## \brief Constructor for DataSeries class
    ##@code
    ## d = DataSeries()
    ## d = DataSeries (rr)
    ##@endcode
    def __init__ (self, handle=None, myData = False):
        if handle == None:
           self._myData = True
           self._data = tpc.telLib.tpCreateTelluriumData(0, 0, "")
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
        if numPyData.dtype.names == None:
            colHdr = []
            for i in range (nrCols):
                colHdr.append ('x' + str (i))
        else:
           colHdr  = numPyData.dtype.names

        columnStr = str(colHdr).strip('[]')
        columnStr = columnStr.strip('()')
        columnStr = columnStr.translate(None, '\' ')
        dataHandle = tpc.telLib.tpCreateTelluriumData(nrRows,nrCols, columnStr)

        #Copy the data
        for row in range(nrRows):
            for col in range(nrCols):
                val = unstructNumPyData[row][col]
                tpc.setTelluriumDataElement(dataHandle, row, col, val)
        return cls(dataHandle, True)

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

    def __getHandle (self):
        return self._data

    # Use x.rows to get the number of rows
    def __getNumberOfRows (self):
        return tpc.getTelluriumDataNumRows(self._data)
    # Use x.toNumpy to get NumPy array
    def __toNumpy (self):
        return tpc.getNumpyData (self._data)

    # Use x.cols to get the number of columns
    def __getNumberOfColumns (self):
        return tpc.getTelluriumDataNumCols(self._data)

    # Use x.toNumpy to get NumPy array
    def __toNumpy (self):
        return tpc.getNumpyData (self._data)

    ## \brief Retrieve the column headers as a list
    ##@code
    ## print d.getColumnHeaders()
    ##@endcode
    def getColumnHeaders (self):
        value = tpc.getTelluriumDataColumnHeader(self._data)
        if value == None:
           value = []
        return value

    ## \brief Get a specific element from a dataseries
    ##@code
    ## print d.getElement (1,2)
    ##@endcode
    def getElement (self, row, col):
        rowCount = tpc.telLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.telLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")

        val = ctypes.c_double()
        if tpc.telLib.tpGetTelluriumDataElement(self._data, row, col, ctypes.byref(val)) == True:
           return val.value
        else:
           # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           # Is there a getLastError for this?
           raise Exception("Unable to retrieve element")

    ## \brief Set a specific element
    ##@code
    ## d.setElement (1,2, 3.1415)
    ##@endcode
    def setElement (self, row, col, value):
        rowCount = tpc.telLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.telLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")
        setTelluriumDataElement(self._data, row, col, value)

    ## \brief Get the weight for a specific data element
    ##@code
    ## print d.getWeight (1,2)
    ##@endcode
    def getWeight (self, row, col):
        rowCount = tpc.telLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.telLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")

        if not tpc.telLib.tpHasWeights(self._data):
            raise Exception("This data object do not have any weights allocated. Allocate weights first before using.")

        val = ctypes.c_double()
        if tpc.telLib.tpGetTelluriumDataWeight(self._data, row, col, ctypes.byref(val)) == True:
           return val.value
        else:
           # Is there a getLastError for this?
           # TK: When an API function fails, the reason for the failure should always be recorded in a message in getLastError()
           msg = tpc.telLib.tpGetLastError()
           raise Exception("Unable to retrieve element. The problem was: {}".format(msg))

    ## \brief Set a specific element
    ##@code
    ## d.setElement (1,2, 3.1415)
    ##@endcode
    def setWeight (self, row, col, value):
        rowCount = tpc.telLib.tpGetTelluriumDataNumRows(self._data)
        colCount = tpc.telLib.tpGetTelluriumDataNumCols(self._data)
        if (row < 0) or (col < 0):
            raise Exception("DataSeries indices must be positive")
        if row >= rowCount:
            raise Exception("Row index out of bounds in dataseries element access")
        if col >= colCount:
            raise Exception("Column index out of bounds in dataseries element access")

        if not tpc.telLib.tpHasWeights(self._data):
            raise Exception("This data object do not have any weights allocated. Allocate weights first before using.")

        tpc.setTelluriumDataWeight(self._data, row, col, value)


    ## \brief Read a dataseries from a file
    ##@code
    ## d.readDataSeries ("myDataSeries.txt")
    ##@endcode
    @classmethod
    def readDataSeries(cls, fileName):
        if not os.path.isfile (fileName):
            raise Exception ("File not found: " + fileName)
        data = tpc.createTelluriumDataFromFile (fileName)
        return cls (data, True)

    ## \brief Write a dataseries to a file
    ##@code
    ## d.writeDataSeries ("myDataSeries.txt")
    ##@endcode
    def writeDataSeries(self, fileName):
        tpc.writeTelluriumData(self._data, fileName)

    ## \brief Plot a dataseries as a graph
    ##@code
    ## d.plot()
    ##@endcode
    def plot (self):
        hdr = tpc.getTelluriumDataColumnHeader(self._data)
        npData = tpc.getNumpyData(self._data)
        tpc.plotTelluriumData(npData, hdr)

    ## \brief Plot a dataseries using Bifurcation additional data as input
    ##@code
    ## d.plot()
    ##@endcode
    def plotBifurcationDiagram (self, bfPoints, bfLabels, legend=True, cmap=None,
                                xlabel=None, ylabel=None, selections=None):
        hdr = tpc.getTelluriumDataColumnHeader(self._data)
        npData = tpc.getNumpyData(self._data)
        tpc.plotBifurcationData(npData, hdr, bfPoints, bfLabels, legend, cmap,
                                xlabel, ylabel, selections)

    data = property (__getHandle)

    ## \brief Return a numpy array from a data series
    ##@code
    ## myarray = d.toNumpy
    ##@endcode
    toNumpy = property (__toNumpy)

    ## \brief Return the number of rows in the data series
    ##@code
    ## print d.rows
    ##@endcode
    rows = property (__getNumberOfRows)

    ## \brief Return the number of columns in the data series
    ##@code
    ## print d.cols
    ##@endcode
    cols = property (__getNumberOfColumns)


# ------------------------------------------------------------------------

class Event(object):

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

class Plugin (object):

    _OnProgressList = Event()
    _onProgressEvent = 0
    _propertyNames = []

    ## \brief Create a Plugin instance
    ##@code
    ## myPlugin = Plugin ("tel_add_noise")
    ##@endcode
    def __init__(self, pluginName):
        self.pluginName = pluginName
        self.plugin = tpc.loadPlugin (_pluginManager, pluginName)
        if self.plugin is None:
            raise Exception(tpc.getLastError())
        else:
            lp = self.listOfProperties()
            for element in lp:
                self._propertyNames.append (element[0])
            tpc.resetPlugin(self.plugin)

    ## \brief Retrieve the plugins version
    ##@code
    ## print plugin.getVersion()
    ##@endcode
    def getVersion(self):
        text = tpc.getPluginVersion(self.plugin)
        return text

    ## \brief Set a given propoerty in the plugin.
    ##@code
    ## myPlugin.setProperty ("Sigma", 0.1)
    ##@endcode
    def setProperty(self, name, value):
        if (isinstance (value, DataSeries)):
           if not tpc.setPluginProperty (self.plugin, name, value.data):
              raise TypeError ("Unable to locate property: ", name)
        else:
           handle  = tpc.getPluginProperty(self.plugin, name);
           if handle == 0:
              raise ValueError ("Unable to locate property: ", name)
           t1 = tpc.getPropertyType (handle)
           if (t1 == "listOfProperties"):
              if isinstance (value, list):
                 if len(value) != 2:
                    raise TypeError ("Expecting two elements in the property list")
                 if not isinstance(value[0], str):
                     raise TypeError("Expecting property name in first element of list")
                 if (not isinstance(value[1], float)) and (isinstance(value[1], int)):
                     raise TypeError("Expecting floating value in second element of list")
                 para1 = tpc.createProperty(value[0], "double", "", value[1])
                 tpc.addPropertyToList (handle, para1)
              else:
                 raise  TypeError ("Expecting a list in setProperty")
           else:
              if not tpc.setPluginProperty (self.plugin, name, value):
                  raise TypeError("Failed to set property {} = {}".format(name, value))

    ## \brief Get the value for a given propoerty in the plugin.
    ##@code
    ## print myPlugin.getProperty("Sigma")
    ##@endcode
    def getProperty (self, name):
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

    ## \brief List all the properties in the plugin
    ##@code
    ## print myPlugin.listOfProperties()
    ##@endcode
    def listOfProperties (self):
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

## \brief List all the property names in the plugin
    ##@code
    ## print myPlugin.listOfPropertyNames()
    ##@endcode
    def listOfPropertyNames(self):
        return tpc.getListOfPluginPropertyNames (self.plugin)

    ## \brief List all the property descriptions in the plugin
    ##@code
    ## print myPlugin.listOfPropertyDescriptions()
    ##@endcode
    ##@code
    ## import pprint
    ## print pprint.pprint (na.listOfProperties())
    ##@endcode
    def listOfPropertyDescriptions (self):
        nameList = tpc.getListOfPluginPropertyNames (self.plugin)
        aList = []
        for i in range (0, len (nameList)):
            name = nameList[i]
            handle = tpc.getPluginProperty(self.plugin, nameList[i])
            descr = tpc.getPropertyDescription(handle)
            aList.append ([name, descr])
        return aList

    ## \brief List all the property hints in the plugin
    ##@code
    ## print myPlugin.listOfPropertyHints()
    ##@endcode
    def listOfPropertyHints (self):
        nameList = tpc.getListOfPluginPropertyNames (self.plugin)
        aList = []
        for i in range (0, len (nameList)):
            name = nameList[i]
            handle = tpc.getPluginProperty(self.plugin, nameList[i])
            descr = tpc.getPropertyHint(handle)
            aList.append ([name, descr])
        return aList

    ## \brief List all the property hints in the plugin
    ##@code
    ## print myPlugin.listOfPropertyHints()
    ##@endcode
    def loadDataSeriesAsNumPy (self, fileName):
        rrDataHandle = tpc.createTelluriumDataFromFile (fileName)
        return tpc.getNumpyData (rrDataHandle)

    def OnProgress (self, f):
        # Make sure garbage collector doens't remove the event pointer
        global _onProgressEvent

        _onProgressEvent =  tpc.NotifyEventEx (f)
        # Pass the address of the self object
        theId = id (self)
        tpc.assignOnProgressEvent(self.plugin, _onProgressEvent, theId, None)

    ## \brief Execute the plugin
    ##@code
    ## print myPlugin.execute()
    ##@endcode
    def execute (self):
        if tpc.executePlugin (self.plugin) == True:
            return True
        else:
            raise Exception(tpc.getLastError())

    def executeEx (self, inThread):
        return tpc.executePluginEx (self.plugin, inThread)

    ## \brief Read all text from a file
    ##@code
    ## print myplugin.readAllText ("myfile.txt")
    ##@endcode
    def readAllText(self, fName):
        file = open(fName, 'r')
        str = file.read()
        file.close()
        return str

    def loadPlugins(self):
        tpc.loadPlugins (self.pluginsManager)

    ## \brief Static method to list all plugins
    ##@code
    ## print Plugin.listOfPlugins()
    ##@endcode
    @staticmethod
    def listOfPlugins():
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

    ## \brief If a plugin has a manual, view it
    ##@code
    ## myPlugin.viewManual()
    ##@endcode
    def viewManual (self):
        tpc.displayPluginManual(self.plugin)

    ## \brief Returns the name of the plugin
    ##@code
    ## print myPlugin.name()
    ##@endcode
    def name (self):
        return tpc.getPluginName(self.plugin)

    ## \brief Returns the description of the plugin
    ##@code
    ## print myPlugin.description()
    ##@endcode
    def description (self):
        return tpc.getPluginDescription(self.plugin)

    ## \brief Returns the hint of the plugin
    ##@code
    ## print myPlugin.hint()
    ##@endcode
    def hint (self):
        return tpc.getPluginHint(self.plugin)


    def info (self):
        return tpc.telLib.tpGetPluginInfo(self.plugin)

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

##\mainpage Working with RoadRunner Plugins
##\section Introduction
## The plugin API allows users to easily access roadrunner plugins. Only three concepts need to be understood:
## 1. Loading a plugin
## 2. Setting and Getting plugin properties
## 3. Executing the plugin
##
## Before using the plugin system the plugin library must be imported using the line:
#@code
##import teplugins as tel
#@endcode
## To load a plugin called "tel_add_noise" use the Python command:
#@code
##p = tel.Plugin ("tel_add_noise")
#@endcode
## The variable p now represents the plugin. Plugins expose a number of properties, these are variables that can be inspected or set. For
## example the add noise plugin has a property called Sigma. To set this to a particular value we would use:
#@code
## p.Sigma = 0.5
#@endcode
## Likewise, properties can be inspected:
#@code
## print p.Sigma
#@endcode
## To list all propreties in a plugin use the method listOfProperties. The following uses call together with pprint to make the output
## more understandable:
#@code
## import pprint
## pprint.pprint (p.listOfProperties())
## [['NoiseType', 'Type of noise (Gaussian = 0, Psychological = 1).'],
## ['Sigma', 'Standard deviation of applied noise'],
## ['InputData', 'Data Series to apply the noise to'],
## ['Progress', 'Indicate progress in (0-100%)']]
#@endcode
##
## To run a plugin so that it carries out its function, use the execute method:
#@code
## p.execute()
#@endcode
## The results from an execute call will either be saved to a file or more likely via properties of the plugin. As a trivial example,
## consider a plugin called "add" that has three properties, x, y and z. When the execute() method is called the plugin will
## take the values stored in x and y, add them together and store the result in z. The following script illustrates how the plugin would be used from Python:
#@code
## import teplugins as tel
## p = tel.Plugin ("add")
## p.x = 1.2
## p.y = 4.5
## p.execute()
## print p.z
#@endcode
##
## The following script illustrates a more substantial example using the add_noise plugin
##
##\include telCreateNoisyDataDoxyPage.py
##
##\section DataSeries
## The plugin system supports a special data type called a Data Series. This is a convenient way to represent rows and colums of
## data. The data type also has the ability to label columns with a text string and to associate each value in the data series with an additional
## value called a weight. In practice the data series will usually store experimental data and the weights will represent a measure
## of undertaintly, perhaps a standard deviation, of the data point. A Data Series can be created using the call:
#@code
## import teplugins as tel
## data = tel.DataSeries()
#@endcode
## Data can be entered into a data series either by loading the data from a specially formated file or from a Python NumPy array. For example:
#@code
## data.readDataSeries ("mydata.txt")
#@endcode
## To read numpy arrays into a data series use the code:
#@code
## import numpy as py
## values = py.array([[1,2],{2,5],[3,7]])
## data = tel.DataSeries.fromNumPy (values)
#@endcode
## The number of rows and columns in a Data Series can be obtained using:
#@code
## print data.rows
## print data.cols
#@endcode
## Currently individual values in a data series can be accessed using the set and get methods:
#@code
## p.setElement (1, 2, 4.567)
## print p.getElement (1, 2)
#@endcode
## Data series can be plotted using the plot method:
#@code
## data.plot()
#@endcode
##The following script is an example of using the add_noise plugin. This plugin takes a data series and add a given amount of Guassian noise
## to all data except the data in the first column.
#@code
##    import teplugins as *
##    p = Plugin ("tel_add_noise")
##    p.viewManual()
##    pl = p.listOfProperties()
##    for item in pl:
##        print item
###
##    p.Sigma = 0.00005
##    series = DataSeries.loadDataSeries ("..\\Examples\\testData.dat")
##    series.plot()
##    p.InputData = series
##    p.execute()
##    p.InputData.plot()
###
##    print "Test Finished"
#@endcode
##
##\section Plugins
# Plugin objects are instanciated using Plugin class. For example to instanciate a plugin called myplugin, we would
# use the code:
#@code
#p = Plugin ("myplugin")
#@endcode
#All interactions with plugins are via plugin properties. Values can be set and retrieved via plugin properties.
#For example, if a plugin has a property sigma, we can assign or access the value using the code:
#@code
#p.sigma = 0.1
#print p.sigma
#@endcode
#Plugins have a single method that can be used to excecute the plugin's functionality:
#@code
#p.execute()
#@endcode
#Once a plugin has been executed, any output from the plugin can be retrieved via properties. Let's
#suppose for example there is a plugin all add, which has three properties called, x, y and result. When executed
#the plugin will take the values in x and y, compute the sum and assign it to result. The plugin can therefore
#be used as follows:
#@code
#p = Plugin("add")
#p.x = 3.4
#p.y = 5.6
#p.execute()
#print p.result
#@endcode
##
##
# \defgroup examples Python Example Scripts
# \brief Scripts illuminating concepts regarding Tellurium Plugins
#
## \example telCreateNoisyData.py
## This example demonstrates the use of the AddNoise plugin
## \image html createNoisyData_output.jpg "Image output using the example below."
##
## \example telLevenbergMarquardt.py
## This example demonstrates parameter minimization, using the Levenberg-Marquardt algorithm.
## Output is shown in the image.
## To run the example, make sure you have some 'Experimental' data available.
##
## \image html lmFit1_output.jpg "Image output using the example below."
