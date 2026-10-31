LLVM dependency
===============

Roadrunner generates its simulation code with LLVM, and is built and tested against LLVM 23.1.
Other LLVM versions are not supported.

Prebuilt binaries
-----------------

Prebuilt LLVM binaries, configured the way roadrunner needs them, are published as zip files on the
`llvm-build releases page <https://github.com/sys-bio/llvm-build/releases>`_. Each zip is named::

	llvm-<version>-<os_and_compiler>-<architecture>-<Release|Debug>.zip

The available platforms are:

	- Windows: ``windows-msvc18`` (Visual Studio 2026) and ``windows-msvc2022`` (Visual Studio 2022), ``x86_64``, Release and Debug
	- macOS: ``macos-14`` (``arm64``) and ``macos-15`` (``x86_64``), Release and Debug
	- Ubuntu: ``ubuntu-24``, ``x86_64``, Release and Debug
	- manylinux: ``manylinux_2_28``, ``x86_64``, Release only

.. note::

	On Windows, build roadrunner with the same Visual Studio version that built the LLVM binaries, or a newer one.
	Linking against binaries built by a newer compiler than yours fails with unresolved ``__std_*`` symbols.

.. note::

	Use the Debug binaries if you build roadrunner in Debug mode. On Windows, Release and Debug binaries cannot be mixed.

After downloading, extract the zip into a folder of its own and remember where you put it. The zip unpacks directly into
``bin``, ``include`` and ``lib`` folders, and the folder that contains them is the path you need as input to building roadrunner
(``LLVM_INSTALL_PREFIX``).

Build from source
-----------------

If no prebuilt binary matches your platform, build LLVM yourself using the recipe in the
`llvm-build <https://github.com/sys-bio/llvm-build>`_ repository. It downloads the official LLVM source release and builds it with the
options roadrunner needs, which are listed in its ``config/llvm-flags.cmake``. Most importantly:

	- RTTI and exception handling must be enabled (``LLVM_ENABLE_RTTI=ON``, ``LLVM_ENABLE_EH=ON``).
	- The ``X86`` and ``AArch64`` targets are built.
	- Static libraries are used (no shared LLVM library).

The result is installed with ``cmake --build . --target install``, and the install folder is the ``LLVM_INSTALL_PREFIX`` for roadrunner.
