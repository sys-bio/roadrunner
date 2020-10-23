import tellurium as te
from rrplugins import *
auto = Plugin("tel_auto2000")


r = te.loada (
	'''   
    	$Xo -> x; 0.1 + k1*x^4/(k2+x^4);    
     	x -> $w; k3*x;
    	// Initialization here    
    	k1 = 0.9;
    	k2 = 0.3;    
    	k3 = 0.7;
	'''
)
 
auto.setProperty("SBML", r.getCurrentSBML())
auto.setProperty("NMX", 5000)
auto.setProperty("ScanDirection", "Positive")
auto.setProperty("PrincipalContinuationParameter", "k3")

auto.setProperty("PCPLowerBound", 0.2)
auto.setProperty("PCPUpperBound", 1.8)
 
#Execute the plugin 
auto.execute() 
 
# Bifurcation summary 
print("Summary: " + auto.BifurcationSummary) 
 
#Plot Bifurcation diagram 
pts     = auto.BifurcationPoints 
lbls    = auto.BifurcationLabels 
biData  = auto.BifurcationData 
 
biData.plotBifurcationDiagram(pts, lbls) 