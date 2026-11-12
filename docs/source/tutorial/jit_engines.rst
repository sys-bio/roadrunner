Jit Engines
==============
Jit stands for just-in-time and a Jit compiler compiles code on-the-fly
into machine code, ready for execution within a single session. RoadRunner
compiles SBML models into machine code for fast evaluation of model rate
laws.

RoadRunner has two Jit backends, MCJit and LLJit. By virtue of
being older, MCJit is more stable than LLJit, which is
a more recent addition to RoadRunner. However, LLJit compiles faster,
and gives the user the option to choose between different levels
of compile time optimization. For those familiar with C and C++,
this is akin to the "-O3" optimization flag.

Given that MCJit is more stable, we have decided to keep it as our default
Jit engine, at least until LLJit has been battle tested. In a future release, however
LLJit will become the default Jit engine while MCJit will be deprecated and then
removed.


Switching Jit Engines
---------------------
Switching backends is remarkably simple. All we need to do is use the `Config.setValue`
function to globally configure the `LLVM_BACKEND` variable.

.. code-block:: python
    :linenos:

    from roadrunner import Config
    Config.setValue(Config.LLVM_BACKEND, Config.LLJIT)

And to switch back it would be:

.. code-block:: python
    :linenos:

    from roadrunner import Config
    Config.setValue(Config.LLVM_BACKEND, Config.MCJIT)

Currently, these are our only two options. But it is foreseeable
that other options will be available in the future.

Modifying LLJit Compile-Time Optimization
-----------------------------------------
There are four optimization levels that you can choose from: `NONE`,
`LESS`, `DEFAULT` and `AGGRESSIVE`. No optimization means faster
model compiles but slower compiled code, since all compiler optimizations
are turned off. On the other end, `AGGRESSIVE` optimization makes for
faster executable models but it takes longer to compile. The default
is `NONE`, as this resulted in 2x faster model load times, with no appreciable
difference seen in most simulation times.  Higher optimization levels may be
at least somewhat effective in some model/simulation combinations; gillespie
simulations seemed to be the most likely to be sped up in our tests by
higher optimization levels.

Here's how to do it:

.. code-block:: python
    :linenos:

    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.NONE)
    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.LESS)
    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.DEFAULT)
    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.AGGRESSIVE)

Modifying LLJit number of threads
-----------------------------------------
LLJIT (but not MCJIT) can be parallelized for model loading.  However,
roadrunner uses only one 'IR module' when compiling, and those modules
are the elements that can be parallelized, so we have seen no speedups
from increasing the number of threads.  In fact, setting the number of
threads to '0' means that compilation is performed on the calling thread,
which in our hands resulted in speedups of 5-35%.  '0 threads' is therefore
the default.

Here's how to do it:

.. code-block:: python
    :linenos:

    Config.setValue(Config.LLJIT_NUM_THREADS, 0)
    Config.setValue(Config.LLJIT_NUM_THREADS, 1)

