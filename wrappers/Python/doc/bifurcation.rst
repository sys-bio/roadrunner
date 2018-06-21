*********************
Bifurcation Analysis
*********************

Bifurcation analysis is useful for understanding model with multiple steady states. RoadRunner supports bifurcation analysis through `rrplugins <https://github.com/sys-bio/rrplugins>`_ package, which is an extension package to RoadRunner and provides an interface to AUTO2000.

Start using rrplugins package by importing the library and creating a Plugin instance. 

     >>> from rrplugins import *
     >>> auto = Plugin("tel_auto2000")

The Plugin instance will have various properties you can set. For example, to load a model:

     >>> sbmlModel = "path/to/SBML/model/"
     >>> auto.setProperty("SBML", readAllText(sbmlModel))

To change number maximum number of steps:

     >>> auto.setProperty("NMX", 5000)

To change the scan direction:

     >>> auto.setProperty("ScanDirection", "Positive")

To change principal continuation parameter as well as its lower and upper bounds:

     >>> auto.setProperty("PrincipalContinuationParameter", A)
     >>> auto.setProperty("PCPLowerBound", 10)
     >>> auto.setProperty("PCPUpperBound", 200)

Execute the plugin by running:

     >>> auto.execute()
     
rrplugins package comes with several plotting functions as well, which is espeicially useful for plotting bifucation diagrams with ease. After executing the plugin, results will be stored in ``BifurcationPoints``, ``BifurcationLabels``, and ``BifurcationData``. You can use these information to plot a bifurcation diagram.

     >>> pts     = auto.BifurcationPoints 
     >>> lbls    = auto.BifurcationLabels 
     >>> biData  = auto.BifurcationData 
     >>> biData.plotBifurcationDiagram(pts, lbls) 

To see the full list of properties, go to `rrplugins AUTO2000 documentation <https://sys-bio.github.io/rrplugins/docs/plugins/auto2000/index.html>`_
