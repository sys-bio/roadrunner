REM @echo off
REM Pushes environment variables
setlocal
REM Allows us to properly set and use values of variables inside of if statements/for loops
setlocal EnableDelayedExpansion

REM Check that they are in the root directory, not in the roadrunner source folder
for %%a in ("%~dp0\.") do set FOLDER=%%~nxa
if "%FOLDER%"=="roadrunner" (
  echo Error: You need to move this script to the rr root folder.
  echo        See the build instructions on the sys-bio/roadrunner Github wiki
  goto usage
)
if not "%FOLDER%"=="rr" (
  echo Warning: Roadrunner root folder should be named rr. 
  echo          Make sure that you are following the build instructions on the Github wiki
)

REM Check for debug or release command line flag
REM Check for generator flag (VS201X or Ninja)
REM Set defaults
set ARCH=x64
set BUILD_TESTS=OFF
set GEN_ARCH=
set GEN=
set CONFIG=
set CONF_SUF=
set LLVM_CONFIG_EXECUTABLE=

REM Check for arch flag
REM Check for llvm-config
REM Check for building tests
REM Check for configuration

if "%1"=="" goto usage

:argloop
if not "%1"=="" (
  if "%1"=="--help" goto usage
  if "%1"=="--build-deps" set BUILD_DEPS=ON
  if "%1"=="--build-llvm" set BUILD_LLVM=ON
  if "%1"=="--build-roadrunner" set BUILD_ROADRUNNER=ON
  if "%1"=="--arch" set ARCH=%2
  if !ARCH!==x64 set GEN_ARCH= Win64
  if "%1"=="--build-tests" set BUILD_TESTS=ON
  if "%1"=="--llvm-config" set LLVM_CONFIG_EXECUTABLE=%2
  REM Generator is required
  if "%1"=="--generator" (
    if "%2"=="Ninja"  set GEN=Ninja
    REM                                       Must put no space between the year and arch
    if "%2"=="VS2015" set GEN="Visual Studio 14 2015!GEN_ARCH!"
    if "%2"=="VS2015" set GEN="Visual Studio 15 2017!GEN_ARCH!"
    if "%2"=="VS2015" set GEN="Visual Studio 16 2019!GEN_ARCH!"
  )
  if "%1"=="--config" (
    set CONFIG=%2
    if "!CONFIG!"=="Debug" (
      set CONF_SUF=_debug
    ) else if "!CONFIG!"=="Release" (
      set CONF_SUF=_release
    ) else (
      echo Invalid configuration: "!CONFIG!"
      goto usage
    )
  )
  SHIFT
  GOTO argloop
)

REM Sanity check input
if "%GEN%"=="" (
  echo Error: Must specify a generator
  goto usage
)
if "%CONFIG%"=="" (
  echo Error: Must specify a configuration
  goto usage
)
if not "%BUILD_DEPS%"=="ON" if not "%BUILD_LLVM%"=="ON" if not "%BUILD_ROADRUNNER%"=="ON" (
  echo Error: Must build one of deps, llvm, or roadrunner
  goto usage
)
REM Check that there is LLVM installed if we're building roadrunner OR that they
REM passed in the llvm-config path
if "%BUILD_ROADRUNNER%"=="ON" (
  if not exist "install%CONF_SUF%\llvm" (
  if "%LLVM_CONFIG_EXECUTABLE%"=="" (
  goto rr_check_done
  :rr_check_err
  echo Error: Building roadrunner, but there is no LLVM installation 
  echo        and llvm-config was not set
  goto usage
)))
:rr_check_done

REM Test what the latest version of Visual Studio they have installed is
REM FIXME

REM Set environment variables for MSVC toolchain
REM Use CALL because it is a batch function
REM Check if DevEnvDir is already defined because calling vcvarsall makes path bigger and windows has a maximum path length
REM FIXME Make this adapt to the version fo Visual Studio they have installed
if not defined DevEnvDir (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" %ARCH%
)

REM Check that they have the appropriate tools installed
if %GEN%==Ninja (
  ninja --version
  if errorlevel 9009 (
    echo Generator was set to Ninja but Ninja was not found
    goto usage
  )
) else (
  REM We should be able to invoke cl because we used vcvarsall earlier
  cl
  if errorlevel 9009 (
    echo Generator was set to MSVC but cl.exe was not found
    goto usage
  )
)

REM If they want to build deps
if not "%BUILD_DEPS%"=="ON" goto deps_done

REM Set up libroadrunner-deps
mkdir build%CONF_SUF%\libroadrunner-deps\
cd build%CONF_SUF%\libroadrunner-deps\
REM Use the && ^ to execute the next line only if this line succeeds
cmake -G %GEN% -DCMAKE_INSTALL_PREFIX=..\..\install%CONF_SUF%\roadrunner ..\..\source\libroadrunner-deps && ^
cmake --build . --config %CONFIG% --target install
REM Exit if there was a failure in the build stage
if %errorlevel% neq 0 (
  echo Could not build and install libroadrunner-deps
  cd ..\..\
  endlocal
  REM Only exit the script, not cmd.exe
  exit /B
)
cd ..\..\
:deps_done

REM If they want to build LLVM from source
if not "%BUILD_LLVM%"=="ON" goto :llvm_done

mkdir build%CONF_SUF%\llvm\
cd build%CONF_SUF%\llvm\

cmake -G %GEN% -DCMAKE_INSTALL_PREFIX=..\..\install%CONF_SUF%\llvm -DLLVM_TARGETS_TO_BUILD=X86 %LLVM_SRC% && ^
cmake --build . --config %CONFIG% --target install

if %errorlevel% neq 0 (
  echo Could not build and install LLVM
  endlocal
  REM Only exit the script, not cmd.exe
  exit /B
)
cd ..\..\
:llvm_done


if not "%BUILD_ROADRUNNER%"=="ON" goto :rr_done
mkdir build%CONF_SUF%\roadrunner
cd build%CONF_SUF%\roadrunner

cmake -G %GEN% -DTHIRD_PARTY_INSTALL_FOLDER=..\..\install%CONF_SUF%\roadrunner -DCMAKE_INSTALL_PREFIX=..\..\install%CONF_SUF%\roadrunner -DLLVM_CONFIG_EXECUTABLE=%LLVM_CONFIG_EXECUTABLE% -DBUILD_TESTS=%BUILD_TESTS% -DBUILD_TEST_TOOLS=%BUILD_TESTS% ..\..\source\roadrunner && ^
cmake --build . --config %CONFIG% --target install

if %errorlevel% neq 0 (
  echo Could not build and install roadrunner
  cd ..\..\
  endlocal
  exit /B
)
cd ..\..\
:rr_done

echo Build and install complete.
exit /B

:usage
  echo Usage
  echo --build-deps
  echo --build-llvm
  echo --build-roadrunner
  echo --arch [x64, x86]                             (Defaults to x64)
  echo --build-tests
  echo --llvm-src    path/to/llvm.src                (Use if you are building LLVM)
  echo --llvm-config path/to/llvm-config.exe         (Use if LLVM is already built)
  echo --generator [Ninja, VS2015, VS2017, VS2019]   (Required)
  echo --config [Debug, Release]                     (Required)
  endlocal
  exit /B 0