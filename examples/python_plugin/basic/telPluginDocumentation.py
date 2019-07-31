#-------------------------------------------------------------------------------
# Purpose: Example demonstrating various documentation features of plugins.
# The plugin shows how to get names of a plugins properties, get description of each property
# and finaly shows how to view a plugins embedded (PDF) manual.  
#
# Author: Totte Karlsson
#-------------------------------------------------------------------------------

from teplugins import *

try:
    pluginName = "levenberg_marquardt"
    plugin = Plugin('tel_' + pluginName)

    #Get some general info about the plugin
    props = plugin.listOfProperties()

    propNr = 1
    for aProperty in props:
        print 'Property ' + `propNr` + '  ' +  `aProperty`
        propNr += 1

    descriptions = plugin.listOfPropertyDescriptions()

    descrpNr = 1
    for aDescr in descriptions:
        print 'Description ' + `descrpNr` + '  ' +  `aDescr`
        descrpNr += 1

    plugin.viewManual()

except Exception as e:
    print 'Problem: ' + `e`
