LLVM-6.x dependency
=====================

Prebuilt binaries
--------------------------

You may use prebuilt binaries if the compiler you use to build roadrunner is the same as the compiler use to build llvm-6.x:

	- `VisualStudio 2019 x64 release <https://github.com/sys-bio/llvm-6.x/releases/download/release%2F6.x/llvm-6.x-msvc2019-x64-release.zip>`_
	- `VisualStudio 2019 x64 debug <https://github.com/sys-bio/llvm-6.x/releases/download/release%2F6.x/llvm-6.x-msvc2019-x64-debug.zip>`_
	- `GCC7.5.0 x64 release <https://github.com/sys-bio/llvm-6.x/releases/download/release%2F6.x/llvm-6.x-gcc7.5.0-x64-release.tar.gz>`_
	- `AppleClang11.0.3 darwin19.4.0 x86-64 release <https://github.com/sys-bio/llvm-6.x/releases/download/release%2F6.x/llvm-6.x-macos-x64-release.zip>`_
	- `AppleClang11.0.3 darwin19.4.0 x86-64 debug <https://github.com/sys-bio/llvm-6.x/releases/download/release%2F6.x/llvm-6.x-macos-x64-debug.zip>`_

.. note::

	The debug binary for llvm-6.x on linux (gcc) is too big for github, so you will have to build this yourself if you want llvm in debug mode on ubuntu.

After downloading llvm-6.x decompress the archive and remember where you put it. You will need this path as input to building roadrunner.

Build from source
-----------------

.. code-block:: bash

	$ git clone https://github.com/sys-bio/llvm-6.x.git
	$ cd llvm-6.x
	$ mkdir build
	$ cd build
	$ cmake -DCMAKE_INSTALL_PREFIX="D:\llvm-6.x\install-msvc2019" -DCMAKE_BUILD_TYPE="Release" ..
	$ cmake --build . --target install --config Release

