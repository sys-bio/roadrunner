import rrplugins

#Create a plugin manager
pm = rrplugins.createPluginManager()

#Load the 'noise' plugin in order to add some noise to the data
plugin = rrplugins.loadPlugin(pm, "tel_add_noise")
if not plugin:
    print(rrplugins.getLastError())
    exit()

print(rrplugins.getPluginInfo(plugin))

#Get a plugins properties as names
print(rrplugins.getListOfPluginPropertyNames(plugin))

paraHandle   = rrplugins.getPluginProperty(plugin, "NoiseType")

print('getting some info about the property')
print(rrplugins.getPropertyInfo(paraHandle))

paraValueString = rrplugins.getPropertyValueAsString(paraHandle)
print('property value is ', paraValueString)

rrplugins.setPropertyByString(paraHandle, "1")
paraValueString = rrplugins.getPropertyValueAsString(paraHandle)
print('Property value is now: ', paraValueString)

rrplugins.setIntProperty(paraHandle, 0)
test = rrplugins.getIntProperty(paraHandle)
print(test)

paraValueString = rrplugins.getPropertyValueAsString(paraHandle)
print('Property value is now: ', paraValueString)

## Retrieve a property handle
paraHandle = rrplugins.getPluginProperty(plugin, "Sigma")

print('getting some info about the property')
print(rrplugins.getPropertyInfo(paraHandle))

paraValueString = rrplugins.getPropertyValueAsString(paraHandle)
print('property is ', paraValueString)

rrplugins.setPropertyByString(paraHandle, "1.23")
paraValueString = rrplugins.getPropertyValueAsString(paraHandle)
print('Property is now: ', paraValueString)

var = 2.23
rrplugins.setDoubleProperty(paraHandle, var)
test = rrplugins.getPropertyValue(paraHandle)
print(test)

paraValueString = rrplugins.getPropertyValueAsString(paraHandle)
print('Property is now: ', paraValueString)

#======= Test bools
boolParaHandle = rrplugins.createProperty("bool", "bool", "Hint", True)
print(rrplugins.getBoolProperty(boolParaHandle))
rrplugins.setBoolProperty(boolParaHandle, False)
print(rrplugins.getBoolProperty(boolParaHandle))

#====Test some strings
stringParaHandle = rrplugins.createProperty("myStringProperty", "string", "Hello", "Hello2")
print(rrplugins.getPropertyValueAsString(stringParaHandle))
# print(rrplugins.getStringProperty(stringParaHandle))
rrplugins.setStringProperty(stringParaHandle, "Setting a string value")
print(rrplugins.getPropertyValueAsString(stringParaHandle))

#====Test list of propertys Todo:: clean this up!!
lOfP = rrplugins.createProperty("test", "listOfProperties", "Hello1")
print('Names in property list:', rrplugins.getNamesFromPropertyList(lOfP))
lOfP2 = rrplugins.createProperty("test", "listOfProperties", "Hello2")

rrplugins.addPropertyToList(lOfP, stringParaHandle)
rrplugins.addPropertyToList(lOfP, stringParaHandle)
rrplugins.addPropertyToList(lOfP2, stringParaHandle)

print('Names in property list:', rrplugins.getNamesFromPropertyList(lOfP))

#Assignment 
rrplugins.setListProperty(lOfP, lOfP2)
print('Names in property list:', rrplugins.getNamesFromPropertyList(lOfP))
print('Names in property list:', rrplugins.getNamesFromPropertyList(lOfP2))



listParaHandle = rrplugins.createProperty("myList", "listOfProperties", "a Hint")

#Should be empty
print(rrplugins.getNamesFromPropertyList(listParaHandle))

#Add a new parameter
aParameterHandle = rrplugins.createProperty("newOne", "double", "the hint")
rrplugins.addPropertyToList(listParaHandle, aParameterHandle)

rrplugins.setDoubleProperty(aParameterHandle, 34)
print(rrplugins.getDoubleProperty(aParameterHandle))

#Or you can use
print(rrplugins.getProperty(aParameterHandle))

#Should not be empty
print(rrplugins.getNamesFromPropertyList (listParaHandle))

#Get the parameter
paraHandle = rrplugins.getFirstProperty(listParaHandle)

print(rrplugins.getPropertyHint (paraHandle))
print(rrplugins.getPropertyInfo (paraHandle))
print(rrplugins.getNamesFromPropertyList (listParaHandle))


print(rrplugins.clearPropertyList(listParaHandle))
#Should be empty
print(rrplugins.getNamesFromPropertyList (listParaHandle))

rrplugins.freeProperty(listParaHandle)

