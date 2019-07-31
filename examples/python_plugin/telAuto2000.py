#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to setup the autot2000 plugin and
# plot a Bifurcation diagram
#
# Author: Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------
from rrplugins import *

sbmlModel ="bistable.xml"                              
auto = Plugin("tel_auto2000")              

#print auto.listOfPropertyNames()

#Setup Auto Properties
auto.setProperty("SBML", readAllText(sbmlModel))

#Auto specific properties
auto.setProperty("ScanDirection", "Negative")    
auto.setProperty("PrincipalContinuationParameter", "k3")
auto.setProperty("PCPLowerBound", .35)
auto.setProperty("PCPUpperBound", 1.5)
    
#Max numberof points
auto.setProperty("NMX", 5000)  
        
#Execute the plugin
auto.execute()

# Bifurcation summary
print "Summary: " + auto.BifurcationSummary

#Plot Bifurcation diagram
pts     = auto.BifurcationPoints
lbls    = auto.BifurcationLabels
biData  = auto.BifurcationData

biData.plotBifurcationDiagram(pts, lbls) 

#auto.viewManual()
print('Plugin version: {}'.format(auto.getVersion()))
