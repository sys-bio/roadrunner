#!/bin/sh

# lame ass quick hack to make the docs
# this 'script' is not intended to live any longer than a 
# week or so utill we get a system down to make docs.

cwd=""`pwd`""

echo "cwd is ${cwd}"


cd "$cwd/source"
doxygen doxygen.conf
cd "$cwd"


#cd "$cwd/wrappers/C"
#doxygen core_c_api.doxy
#cd "$cwd"

cd  "$cwd/wrappers/Python/doc"
make html
cd "$cwd"

