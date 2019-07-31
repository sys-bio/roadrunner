#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to setup the autot2000 plugin and
# plot a Bifurcation diagram
#
# Author:      Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------
from rrplugins import *
import roadrunner as rr

try:  
    sbmlModel ="irreversible_bistability.xml"                              
    
    #rrm = rr.RoadRunner(sbmlModel)
    #result = rrm.simulate()
    #rrm.plot()
    
    auto = Plugin("tel_auto2000")              
    
    #print auto.listOfPropertyNames()
    
    #Setup Auto Propertys
    auto.setProperty("SBML", readAllText(sbmlModel))
    
    #Auto specific properties
    auto.setProperty("ScanDirection", "Positive")    
    auto.setProperty("PrincipalContinuationParameter", "Signal")
    auto.setProperty("PCPLowerBound", -3)
    auto.setProperty("PCPUpperBound", 4)
    auto.setProperty("PreSimulation", True)
    #auto.setProperty("PreSimulationStart", 0)
    #auto.setProperty("PreSimulationDuration", 100)
    #auto.setProperty("PreSimulationSteps", 100)
    
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
    
    print "Done"

except Exception as e:
    print "There was a problem: " + `e`
