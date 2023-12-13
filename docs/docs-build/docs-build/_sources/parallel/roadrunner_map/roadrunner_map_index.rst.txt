Parallel RoadRunnerMap
=======================

RoadRunner compiles sbml code into machine code for fast execution of
model simulations. However model compilation takes time and can be prohibitory
if users need to simulate a large variety of models at once. To alleviate
some of this burden, we provide a `RoadRunnerMap` object, which is a container
for roadrunner objects. The `RoadRunnerMap` has a `dict`-like interface and is intended
to be a drop in replacement for a python dictionary specifically for use with
roadrunner models. The major difference is that `RoadRunnerMap` thread-safe
and uses a threadpool for building and storing models. The interested reader
can refer to the open source `parallel hashmap <https://github.com/greg7mdp/parallel-hashmap>`_
for the thread-safe hash map and `thread-pool <https://github.com/bshoshany/thread-pool>`_
for the multithreading capabilities used by `RoadRunnerMap`.

.. note:

    RoadRunnerMap is an experimental feature. Please report any irregularities
    or bugs you might find by opening a `github issue <https://github.com/sys-bio/roadrunner/issues>`_


For these demonstrations, we require some sbml models to load. To simplify
this process we provide a function that will download the curated section
of biomodels locally for you to use.

.. note::

    Windows likes to hang on to open files, meaning we cannot clean up
    all trace files used in the download process. You can delete any residual
    manually.

Here is the code for downloading the curated section of biomodels. It is imported
in the following examples.

.. literalinclude:: get_biomodels.py
    :linenos:
    :language: python

Loading Models: 1 Thread
-------------------------
When a single thread is requested, serial algorithms take over from the parallel ones.
This avoids some of the overhead required to set up multiple threads and
is useful when you only have a few models to manage.

.. literalinclude:: roadrunner_map_serial.py
    :linenos:
    :language: python

Loading Models: Multithreading
------------------------------
To use multithreading, pass the number of threads you want to use
in to the `RoadRunnerMap` constructor.

.. warning:

    Do not attempt to use too many threads. No attempt is made in `RoadRunnerMap`
    to prevent this so it is up to the user to make a sensible choice, given their
    system. Use of too many threads is called thread over-subscription and will
    slow your program. On a system with 8 physical cores and 16 hyper-threaded,
    you do it would be wise not to exceed 16 threads.


.. literalinclude:: roadrunner_map_3_threads.py
    :linenos:
    :language: python

RoadRunnerMap has a dict-like interface
-----------------------------------------
Many of the functions you are used to from Python's `dict` object
are also available in `RoadRunnerMap`. The
`RoadRunnerMap.keys()`, `RoadRunnerMap.values()` and
`RoadRunnerMap.items()` functions all have linear complexity
O(N) as the map is iterated over to construct lists at runt time.

.. literalinclude:: roadrunner_map_interface.py
    :linenos:
    :language: python

Inserting new models
------------------------
Insertion of a new model into the dictionary works much like you might expect,
except that instead of passing in a fully constructed RoadRunner model,
you instead pass in the sbml string or the path to sbml file. Much like Python's
`dict` object, keys are unique, so inserting another model with the same key
as an existing model will overwrite the old model.

At present, the sbml model name is used as the default key for the model.
We have implemented this as a "first pass" for simplicity but can foresee
problems with model names not being unique. We therefore anticipate improvements
in future releases once user feedback has been generated. For now, users
may specify their own key manually.

.. literalinclude:: roadrunner_map_insertion.py
    :linenos:
    :language: python

