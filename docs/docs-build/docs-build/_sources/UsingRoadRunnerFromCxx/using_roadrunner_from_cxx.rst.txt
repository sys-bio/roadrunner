Using RoadRunner from C++
============================
Despite RoadRunner's user interface being primarily in Python,
RoadRunner is first and foremost a C++ library. It is therefore is possible to
link roadrunner into an executable or library.

.. note::

    You will need to have already installed RoadRunner into a directory on your system.
    See :ref:`Installing RoadRunner` section.

.. note::

    You will also need to install `cmake <https://cmake.org/download/>`_ and add its
    "bin" directory to the PATH environment variable. You will know this is done
    if the command `cmake --help` works on the terminal/powershell.

The following is a very simple example of how to link
roadrunner into an executable and run a simulation from C++.
Create a new folder with whatever name seems appropriate and
create the following files:

.. code-block:: bash

    .
    ├── CMakeLists.txt
    └── main.cpp

The `CMakeLists.txt` file should contain the following:

.. literalinclude:: CMakeLists.txt
    :linenos:
    :language: cmake

Now you can use RoadRunner from C++.

.. literalinclude:: main.cpp
    :linenos:
    :language: C++
