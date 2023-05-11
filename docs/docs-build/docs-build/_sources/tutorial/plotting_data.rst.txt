Plotting Data
===========================

RoadRunner has a built in :meth:`~RoadRunner.plot()` method which can perform basic plotting.
Simply call::

   result = rr.simulate(0, 10, 100)
   rr.plot()

If one wants more control over the data plots, one may use matplotlib directly.  Assuming the
simulate returns an array called result, and that the first column represents the x axis and the
remaining columns the y axis, we type::

   import pylab
   pylab.plot (result[:,0],result[:,1:])
   pylab.show()

This will bring up a new window showing the plot. To clear the plot for next time, type the command::

   pylab.clf()

One may also override the built-in :meth:`~RoadRunner.plot()` method with a more more capable plotting routine.

Below is a simplified version of the :meth:`~RoadRunner.plot()` method. You may copy and write a
customized version and even attach it to the RoadRunner object. The first argument is a RoadRunner
object instance, and the second is a flag which tells the method to show the plot or not::

  def plot(rr, show=True):

      import pylab as p

      result = self.getSimulationData()

      if result is None:
          raise Exception("no simulation result")

      # assume result is a standard numpy array

      selections = rr.timeCourseSelections

      if len(result.shape) != 2 or result.shape[1] != len(selections):
          raise Exception("simulation result columns not equal to number of selections,"
                          "likely a simulation has not been run")

      times = result[:,0]

      for i in range(1, len(selections)):
          series = result[:,i]
          name = selections[i]
          p.plot(times, series, label=str(name))

          p.legend()

      if show:
          p.show()


You can attach your plotting function to the RoadRunner object by simply setting the plot
method::

  def my_plot(rr, show):
      pass

  import roadrunner
  roadrunner.RoadRunner.plot = my_plot

Now, whenever the :meth:`~RoadRunner.plot()` method is called, your plot function will be the one that is
invoked.
