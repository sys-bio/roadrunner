Loading Models
===========================

RoadRunner reads models using the SBML format. If you have a SBML model stored on your hard drive, it is
possible to load that model either by giving he document contents or path to the Roadrunner
constructor, or later by using the method, :meth:`~RoadRunner.load()`. Let's assume you have a model called
mymodel.xml in ``C:\MyModels``. To load this model in **Windows** we would use the command::

   rr = roadrunner.RoadRunner("C:/MyModels/mymodel.xml")

Note, Windows typically used the back slash, '\\' to indicate a directory separator, however in
most languages including python, this is the escape character, therefore one can also enter Windows
paths using the forward slash '/' which does not cause issues. If one want to use the backslash,
these must be typed twice, i.e. ``'C:\\MyModels\\mymodel.xml'``.

On the **Mac or Linux** we might use::

   rr = roadrunner.RoadRunner("/home/MyModels/mymodel.xml")

RoadRunner can also load models directly from a URL via::

   rr = roadrunner.RoadRunner("https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000010.2?filename=BIOMD0000000010_url.xml")

If the model was loaded successfully, the RoadRunner object is now ready to use,  otherwise an exception will
be raised that contains extended information detailing exactly what failed. If any warnings are
found in the SBML document, these will be displayed in the console error log.

It is also possible to load a model from a string containing a SBML model, for example:

   rr = roadrunner.RoadRunner (sbmlStr)

This is useful when one wishes to create a new roadrunner instance from an existing model, eg:

   sbmlStr = rr.getCurrentSBML()
   rrnew = roadrunner.RoadRunner (sbmlStr)

Additionally, there are a couple models **included with libRoadRunner**. The models  ``feedback.xml``
and ``Test_1.xml`` are available in the ``roadrunner.tests`` module. To access these use::

   import roadrunner.tests as test
   rr = test.getRoadRunner('feedback.xml')

There are a few additional models in the ``models/`` directory of the distribution, where you installed libRoadRunner.
