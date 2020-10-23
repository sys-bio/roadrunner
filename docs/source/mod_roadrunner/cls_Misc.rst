Miscellaneous Functions
_______________________

.. py:function:: getCopyrightStr()
   :module: roadrunner

   Returns the copyright string

   
.. py:function:: getVersionStr()
   :module: roadrunner

   Returns the version string
   
   
.. staticmethod:: RoadRunner_getExtendedVersionInfo()
   :module: roadrunner

   :func:`getVersionStr()` with infomation about dependent libs versions.


.. staticmethod:: RoadRunner_getParamPromotedSBML(*args)
   :module: roadrunner

   Takes an SBML document (in textual form) and changes all of the local parameters
   to be global parameters.

   :param str SBML: the contents of an SBML document
   :rtype: str   
   
   
.. py:function:: validateSBML(*args)   
   :module: roadrunner

   Given a string, check whether the string is a valid SBML string. 
   Raises exception if the string is not a valid SBML string.
   
   :param str SBML: the contents of an SBML document
   
   
.. py:function:: listTestModels()
   :module: roadrunner

   Lists all the test models.
   

.. py:function:: loadTestModel(name)
   :module: roadrunner

   Load a test model given a name.
   
   :param str name: name of the test model
   
   
   
   