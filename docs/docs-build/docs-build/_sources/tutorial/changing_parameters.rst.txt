Changing Parameters
===========================

Often during a modeling experiment we will need to change parameter values, that is the values of the various
kinetic constants in the model. If a model has a kinetic constants k1, then we can change or inspect the value using
the following syntax::

   print rr.k1
   rr.k1 = 1.2