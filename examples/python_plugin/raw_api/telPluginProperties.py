from teplugins import *

#Create a plugin manager
pm = createPluginManager()

#Load the 'noise' plugin in order to add some noise to the data
plugin = loadPlugin(pm, "tel_add_noise")
if not plugin:
    print getLastError()
    exit()

print getPluginInfo(plugin)

#Get a plugins properties as names
print getListOfPluginPropertyNames(plugin)

paraHandle   = getPluginProperty(plugin, "NoiseType")

print 'getting some info about the property'
print getPropertyInfo(paraHandle)

paraValueString = getPropertyValueAsString(paraHandle)
print 'property value is ' + paraValueString

setPropertyByString(paraHandle, "1")
paraValueString = getPropertyValueAsString(paraHandle)
print 'Property value is now: ' + paraValueString

setIntProperty(paraHandle, 0)
test = getIntProperty(paraHandle)
print test

paraValueString = getPropertyValueAsString(paraHandle)
print 'Property value is now: ' + paraValueString

## Retrieve a property handle
paraHandle = getPluginProperty(plugin, "Sigma")

print 'getting some info about the property'
print getPropertyInfo(paraHandle)

paraValueString = getPropertyValueAsString(paraHandle)
print 'property is ' + paraValueString

setPropertyByString(paraHandle, "1.23")
paraValueString = getPropertyValueAsString(paraHandle)
print 'Property is now: ' + paraValueString

var = 2.23
setDoubleProperty(paraHandle, var)
test = getPropertyValue(paraHandle)
print test

paraValueString = getPropertyValueAsString(paraHandle)
print 'Property is now: ' + paraValueString

#======= Test bools
boolParaHandle = createProperty("bool", "bool", "Hint", True)
print getBoolProperty(boolParaHandle)
setBoolProperty(boolParaHandle, False)
print getBoolProperty(boolParaHandle)

#====Test some strings
stringParaHandle = createProperty("myStringProperty", "string", "Hello", "Hello2")
print getStringProperty(stringParaHandle)
setStringProperty(stringParaHandle, "Setting a string value")
print getStringProperty(stringParaHandle)

#====Test list of propertys Todo:: clean this up!!
lOfP = createProperty("test", "listOfProperties", "Hello1")
print 'Names in property list:' + `getNamesFromPropertyList(lOfP)`
lOfP2 = createProperty("test", "listOfProperties", "Hello2")

addPropertyToList(lOfP, stringParaHandle)
addPropertyToList(lOfP, stringParaHandle)
addPropertyToList(lOfP2, stringParaHandle)

print 'Names in property list:' + `getNamesFromPropertyList(lOfP)`

#Assignment 
setListProperty(lOfP, lOfP2)
print 'Names in property list:' + `getNamesFromPropertyList(lOfP)`
print 'Names in property list:' + `getNamesFromPropertyList(lOfP2)`



listParaHandle = createProperty("myList", "listOfProperties", "a Hint")

#Should be empty
print getNamesFromPropertyList(listParaHandle)

#Add a new parameter
aParameterHandle = createProperty("newOne", "double", "the hint")
addPropertyToList(listParaHandle, aParameterHandle)

setDoubleProperty(aParameterHandle, 34)
print getDoubleProperty(aParameterHandle)

#Or you can use
print getProperty(aParameterHandle)

#Should not be empty
print getNamesFromPropertyList (listParaHandle)

#Get the parameter
paraHandle = getFirstProperty(listParaHandle)

print getPropertyHint (paraHandle)
print getPropertyInfo (paraHandle)
print getNamesFromPropertyList (listParaHandle)


print clearPropertyList(listParaHandle)
#Should be empty
print getNamesFromPropertyList (listParaHandle)

freeProperty(listParaHandle)

