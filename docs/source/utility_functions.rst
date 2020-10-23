*****************
Utility Functions
*****************

.. currentmodule:: RoadRunner

The following methodsprovide basic information about the current roadRunner installation.


Information Group
-----------------

:meth:`~RoadRunner.getInfo` returns useful information about libRoadRunner's current state::

   >>> print rr.getInfo()
   Model Loaded: false
   ConservationAnalysis: 0
   libSBML version: LibSBML Version: 5.12.0
   Temporary folder: not used
   Compiler location: not used
   Support Code Folder: not used
   Working Directory: C:\dev\eclipse


:meth:`~roadrunner.getVersionStr` returns specific information Strings based on arguments passed to it. 
For example to get the libSBML version used, you can::

   >>> import roadrunner

   >>> print roadrunner.getVersionStr(roadrunner.VERSIONSTR_LIBSBML)
   LibSBML Version: 5.9.0

or you can do any combination of the *VERSIONSTR_* options::

   >>> print roadrunner.getVersionStr(
                                roadrunner.VERSIONSTR_BASIC | 
                                roadrunner.VERSIONSTR_COMPILER | 
                                roadrunner.VERSIONSTR_DATE | 
                                roadrunner.VERSIONSTR_LIBSBML)
   1.4.3; Compiler: Microsoft Visual Studio 2013; Date: Dec 18 2013, 22:59:30; LibSBML Version: 5.12.0

There is also the standard python ``__version__`` which is actually set 
when the module is loaded from ``getVersionStr``::
   
   >>> print roadrunner.__version__
   1.4.3; Compiler: Microsoft Visual Studio 2013; Date: Dec 18 2013, 22:59:30


.. autosummary::

   RoadRunner.getInfo

