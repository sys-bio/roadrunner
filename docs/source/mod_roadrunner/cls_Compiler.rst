JIT Compilation
_______________


.. class:: Compiler()
   :module: RoadRunner

   The Compiler object provides information about the JIT compiler
   currently in use. 

.. method:: Compiler.getCompiler()
   :module: RoadRunner

   gets the name of the JIT compiler, i.e. "LLVM" means we are using the
   LLVM JIT compiler.

   :rtype: str


.. method:: Compiler.getVersion()
   :module: RoadRunner:

   get the version of the JIT compiler.

   :rtype: str


.. method:: Compiler.getDefaultTargetTriple()
   :module: RoadRunner

   Return the default target triple the compiler  has been configured to produce code for.
   A 'triple' is just a string that contains three items, it is called 'triple' as 
   that is a LLVM historical convention.

   The target triple is a string in the format of:
   CPU_TYPE-VENDOR-OPERATING_SYSTEM
   
   or
   
   CPU_TYPE-VENDOR-KERNEL-OPERATING_SYSTEM

   :rtype: str
   
  
.. method:: Compiler.getProcessTriple()
   :module: RoadRunner
            
   Return an appropriate target triple for generating
   code to be loaded into the current process, e.g. when using the JIT.

   :rtype: str


.. method:: Compiler.getHostCPUName()
   :module: RoadRunner

   getHostCPUName - Get the LLVM name for the host CPU. The particular format
   of the name is target dependent, and suitable for passing as -mcpu to the
   target which matches the host.
     
   return - The host CPU name, or empty if the CPU could not be determined.

   :rtype: str
     
.. method:: Compiler.setCompiler(compiler)
   :module: RoadRunner

   A legacy method that currently does not do anything.

.. method:: Compiler.getCompilerLocation()
   :module: RoadRunner

   A legacy method that currently does not do anything.

.. method:: Compiler.setCompilerLocation(loc)
   :module: RoadRunner

   A legacy method that currently does not do anything.

.. method:: Compiler.getSupportCodeFolder()
   :module: RoadRunner

   A legacy method that currently does not do anything.

.. method:: Compiler.setSupportCodeFolder(path)
   :module: RoadRunner

   A legacy method that currently does not do anything.
