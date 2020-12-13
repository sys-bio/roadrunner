#!/usr/bin/env bash
# use only in azure pipelines, where the variables that are needed are defined.
# arguments:
#   llvm_install_prefix: required.
llvm_install_prefix=$1
echo "current driectory is: $(pwd)"
echo "ls $(ls)"
wget -q https://github.com/sys-bio/llvm-6.x/releases/download/release%2F6.x/llvm-6.x-gcc7.5-x64-release.tar.gz
tar -zxf llvm-6.x-gcc7.5-x64-release.tar.gz
mv llvm-6.x-gcc7.5-x64-release/** '$llvm_install_prefix'
echo "cd to LLVM_INSTALL_PREFIX: $llvm_install_prefix"
cd $llvm_install_prefix
echo "ls:"
ls