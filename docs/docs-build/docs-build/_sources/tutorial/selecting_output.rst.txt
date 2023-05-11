Selecting Simulation Output
===========================

RoadRunner supports a range of options for selecting what data a simulation should return. For more
detailed information on selections, see the :ref:`selecting-values` section.

The simulate method, by default returns an `structured array
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_,
which are arrays that also contain column names. These can be plotted directly using the
built in :meth:`~RoadRunner.plot()` function.

The output selections default to time and the set of floating species.
It is possible to change the simulation result values by changing the selection list.
For example assume that a model has three species, S1, S2, and S3 but we only want :meth:`~RoadRunner.simulate()` to
return time in the first column and S2 in the second column. To specify this we would type::

   rr.timeCourseSelections = ['time', 'S2']
   result = rr.simulate (0, 10, 100)

In another example let say we wanted to plot a phase plot where S1 is plotted against S2. To do this we
type the following::

   rr.timeCourseSelections = ['S1', 'S2']
   result = rr.simulate(0, 10, 100)

Some additional examples include:

.. code-block:: Python

   # Select time and two rates of change (dS1/dt and dS2/dt)
   rr.timeCourseSelections = ["time", "S1'", "S2'"]

   # By default species names yield amounts, concentrations can be obtained
   # using square brackets, e.g.
   rr.timeCourseSelections = ['time', '[S1]', '[S2]']

