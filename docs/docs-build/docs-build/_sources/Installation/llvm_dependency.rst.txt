LLVM-13.x dependency
=====================

Prebuilt binaries
--------------------------

You may use prebuilt binaries if the compiler you use to build roadrunner is the same as the compiler use to build llvm-13.x:

	- `VisualStudio 2022 x64 release <https://github.com/sys-bio/llvm-13.x/releases/download/llvmorg-13.0.0/llvm-13.x-msvc2022-x64-rel.zip>`_
	- `VisualStudio 2022 x64 debug <https://github.com/sys-bio/llvm-13.x/releases/download/llvmorg-13.0.0/llvm-13.x-win32-dbg.zip>`_
	- `GCC10 x64 release <https://github.com/sys-bio/llvm-13.x/releases/download/llvmorg-13.0.0/llvm13-ubuntu-gcc10-rel.tar.gz>`_
	- `AppleClang 13 Darwin universal binaries Release <https://github.com/sys-bio/llvm-13.x/releases/download/llvmorg-13.0.0/llvm-13.x-clang13-universal-binaries-rel.zip>`_
	- `AppleClang 13 Darwin Release <https://github.com/sys-bio/llvm-13.x/releases/download/llvmorg-13.0.0/llvm-13.x-Darwin-Release.zip>`_

.. note::

	The debug binary for llvm-13.x on linux (gcc) is too big for github, so you will have to build this yourself if you want llvm in debug mode on ubuntu.

After downloading llvm-13.x decompress the archive and remember where you put it. You will need this path as input to building roadrunner.

Build from source
-----------------

.. code-block:: bash

	$ git clone https://github.com/sys-bio/llvm-13.x.git
	$ cd llvm-13.x
	$ mkdir build
	$ cd build
	$ cmake -DCMAKE_INSTALL_PREFIX="D:\llvm-13.x\install-msvc2019" -DCMAKE_BUILD_TYPE="Release" ../llvm
	$ cmake --build . --target install --config Release

