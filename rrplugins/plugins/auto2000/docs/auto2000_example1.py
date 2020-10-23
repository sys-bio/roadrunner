from rrplugins import *
auto = Plugin("tel_auto2000")


sbmlModel = "path to any model. Eg. some models are provided in source code, like, BIOMD0000000203.xml"

auto.setProperty("SBML", readAllText(sbmlModel))

auto.setProperty("NMX", 5000)

auto.setProperty("ScanDirection", "Positive")

auto.setProperty("PrincipalContinuationParameter", "A")
auto.setProperty("PCPLowerBound", 10)
auto.setProperty("PCPUpperBound", 200)

#Max numberof points 
auto.setProperty("NMX", 5000) 
 
#Execute the plugin 
auto.execute() 
 
# Bifurcation summary 
print("Summary: " + auto.BifurcationSummary) 
 
#Plot Bifurcation diagram 
pts     = auto.BifurcationPoints 
lbls    = auto.BifurcationLabels 
biData  = auto.BifurcationData 
 
biData.plotBifurcationDiagram(pts, lbls) 