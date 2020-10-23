Configuration
_____________

Many of RoadRunner classes use a number of configuration parameters. Most of these can be set using
the Config class. The values stored in the Config class only determine the *default* values of
parameters. The Config class will look in the following locations for the config file, and will
load the values from the first config file it finds. If it does not find a config file in one of
the following locations, a default set of configuration parameters are used. The search locations
of the config file are:

#1: the ROADRUNNER_CONFIG environment variable

#2: try the user's home directory for roadrunner.conf, i.e.::
  
  /Users/andy/roadrunner.conf

#3: try the user's home directory for .roadrunner.conf, i.e.::
  
  /Users/andy/.roadrunner.conf

#4: try the same directory as the roadrunner shared library, this will be the same directory as the python _roadrunner.pyd
python extension module, i.e.::
  
  /Users/andy/local/lib/roadrunner.conf

#5: try one directory up from the where the shared library or program is at, i.e.::
  
  /Users/andy/local/roadrunner.conf

The conf file is just a plain text file of where each line may be key / value pair separated by a
":", i.e. ::

  KEY_NAME : Value

Any line that does not match this format is ignored, and keys that are not found are also
ignored. Therefore, any line that does not start with a word character is considered a comment.

All of the configuration management functions are static method of the Config class, and 
all of the configuration keys are static attributes of the Config class, these are documented in the 
Configuration Functions section. 

As all of the Config class methods are static, one never instantiates the Config class. 

Configuration Functions
-----------------------

.. staticmethod:: Config.setValue(key, value)
   :module: RoadRunner

   Set the value of a configuration key. The value must be either a string, integer, double or
   boolean. If one wanted to turn off moiety conservation (this will not have an effect on
   already loaded models)::

     from roadrunner import Config
     Config.setValue(Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, False)


   Or, other options may be set to Boolean or integer values. To enable an optimization features::

     Config.setValue(Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, True)


.. staticmethod:: Config.getConfigFilePath()
   :module: RoadRunner

   If roadrunner was able to find a configuration file on the file system, its full path is returned
   here. If no file was found, this returns a empty string. 


.. staticmethod:: Config.readConfigFile(path)
   :module: RoadRunner

   Read all of the values from a configuration file at the given path. This overrides any previously
   stored configuration. This allows users to have any number of configuration files and load them
   at any time. Say someone had to use Windows, and they had a file in their C: drive, this would be
   loaded via::
     
     Config.readConfigFile("C:/my_config_file.txt")

   Note, the forward slash works on both Unix and Windows, using the forward slash eliminates the
   need to use a double back slash, "\\\\".

.. staticmethod:: Config.writeConfigFile(path)
   :module: RoadRunner

   Write all of the current configuration values to a file. This could be written to one of the
   default locations, or to any other location, and re-loaded at a later time. 



Available Configuration Parameters
----------------------------------
All of the configuration parameter keys are static attributes of the Config class and are listed
here. The variable type of the parameter is listed after the key name. 


.. attribute:: Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES
   :module: RoadRunner
   :annotation: bool

   Perform conservation analysis. By default, this attribute is set as False.
      
   This causes a re-ordering of the species, so results generated
   with this flag enabled can not be compared index wise to results
   generated otherwise.

   Moiety conservation is only compatible with simple models which do NOT have any events or rules
   which define or alter any floating species, and which have simple constant stoichiometries. 

   Moiety conservation may cause unexpected results, be aware of what it is before enabling. 

   Not recommended for time series simulations.
   
   To enable, type:
   
   >>> roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, True)  


.. attribute:: Config.LOADSBMLOPTIONS_RECOMPILE
   :module: RoadRunner
   :annotation: bool

   Should the model be recompiled?
   The LLVM ModelGenerator maintains a hash table of currently running
   models. If this flag is NOT set, then the generator will look to see
   if there is already a running instance of the given model and
   use the generated code from that one.
  
   If only a single instance of a model is run, there is no
   need to cache the models, and this can safely be enabled,
   realizing some performance gains.


.. attribute:: Config.LOADSBMLOPTIONS_READ_ONLY
   :module: RoadRunner
   :annotation: bool

   If this is set, then a read-only model is generated. A read-only
   model can be simulated, but no code is generated to set model
   values, i.e. parameters, amounts, values, etc...
  
   It takes a finite amount of time to generate the model value setting
   functions, and if they are not needed, one may see some performance
   gains, especially in very large models.


.. attribute:: Config.LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS
   :module: RoadRunner
   :annotation: bool

   Generate accessors functions to allow changing of initial
   conditions.



.. attribute:: Config.LOADSBMLOPTIONS_OPTIMIZE_GVN
   :module: RoadRunner
   :annotation: bool

   GVN - This pass performs global value numbering and redundant load
   elimination contemporaneously.




.. attribute:: Config.LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION
   :module: RoadRunner
   :annotation: bool

   CFGSimplification - Merge basic blocks, eliminate unreachable blocks,
   simplify terminator instructions, etc...




.. attribute:: Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING
   :module: RoadRunner
   :annotation: bool

   InstructionCombining - Combine instructions to form fewer, simple
   instructions. This pass does not modify the CFG, and has a tendency to make
   instructions dead, so a subsequent DCE pass is useful.




.. attribute:: Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION
   :module: RoadRunner
   :annotation: bool

   DeadInstElimination - This pass quickly removes trivially dead instructions
   without modifying the CFG of the function.  It is a BasicBlockPass, so it
   runs efficiently when queued next to other BasicBlockPass's.




.. attribute:: Config.LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION
   :module: RoadRunner
   :annotation: bool

   DeadCodeElimination - This pass is more powerful than DeadInstElimination,
   because it is worklist driven that can potentially revisit instructions when
   their other instructions become dead, to eliminate chains of dead
   computations.


.. attribute:: Config.LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER
   :module: RoadRunner
   :annotation: bool


   InstructionSimplifier - Remove redundant instructions.


.. attribute:: Config.LOADSBMLOPTIONS_USE_MCJIT
   :module: RoadRunner
   :annotation: bool

   Currently disabled. 

   Use the LLVM MCJIT JIT engine.
  
   Defaults to false.
  
   The MCJIT is the new LLVM JIT engine, it is not as well tested as the
   original JIT engine. Does NOT work on LLVM 3.1


.. attribute:: Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES
   :module: RoadRunner
   :annotation: int

   RoadRunner by default dynamically generates accessors properties
   for all SBML symbol names on the model object when it is retrieved
   in Python. This feature is very nice for interactive use, but
   can slow things down. If this feature is not needed, it
   can be disabled here.
         

.. attribute:: Config.ROADRUNNER_DISABLE_WARNINGS
   :module: RoadRunner
   :annotation: int

   Disable SBML conserved moiety warnings.

   Conserved Moiety Conversion may cause unexpected behavior, be aware of what it
   is before enabling. 

   RoadRunner will issue a warning in steadyState if conservedMoieties are NOT 
   enabled because of a potential singular Jacobian. To disable this warning, 
   set this value to 1

   A notice will be issued whenever a document is loaded and conserved moieties 
   are enabled. To disable this notice, set this value to 2.

   To disable both the warning and notice, set this value to 3

   Rationale for these numbers: This is actual a bit field, disabling the steady state 
   warning value is actually 0b01 << 0 which is 1, and the loading warning is 0b01 << 1 
   which is 2 and 0b01 & 0b10 is 0b11 which is 3 in decimal. 


.. attribute:: Config.LOADSBMLOPTIONS_PERMISSIVE
   :module: RoadRunner
   :annotation: int

   Accept some non-valid SBML (such as Booleans in numberic expressions).

   For legacy code only. Do not use.


.. attribute:: Config.MAX_OUTPUT_ROWS
   :module: RoadRunner
   :annotation: int

   Set the maximum number of rows in the output matrix.

   For models with very fine time stepping, the output of simulate can
   use up all available memory and crash the system. This option provides
   an upper bound on the maximum number of rows the output can contain.
   The simulation will be aborted and the output truncated if this value is exceeded.


.. attribute:: Config.ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS
   :module: RoadRunner
   :annotation: bool

   Enable or disable steady state calculations when a model contains events

   If true, steady state calculations will be carried out irrespective 
   of whether events are present or not.
   If false, steady state calculations will not be carried out 
   in the presence of events.



