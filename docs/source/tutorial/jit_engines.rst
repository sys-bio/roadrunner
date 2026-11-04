Jit Engines
==============
Jit stands for just-in-time and a Jit compiler compiles code on-the-fly
into machine code, ready for execution within a single session. RoadRunner
compiles SBML models into machine code for fast evaluation of model rate
laws.

RoadRunner has two Jit backends, MCJit and LLJit. LLJit is the default.
It compiles faster than MCJit, and gives the user the option to choose
between different levels of compile time optimization. For those familiar
with C and C++, this is akin to the "-O2" optimization flag.

MCJit, the older backend, is still available, but will be deprecated and then
removed in a future release.


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
faster executable sbml models but it takes longer to compile. The default
is `DEFAULT`. For a model with 500 species, it compiles about 40% slower
than `NONE`, and the compiled model simulates about four times faster.

Here's how to do it:

.. code-block:: python
    :linenos:

    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.NONE)
    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.LESS)
    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.DEFAULT)
    Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.AGGRESSIVE)





