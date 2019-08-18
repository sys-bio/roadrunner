
from distutils.core import setup, Extension 
#name of module 
name  = "telplugins"
  
#version of module 
version = "1.0"
  
# specify the name of the extension and source files 
# required to compile this 
ext_modules = Extension(name='_telplugins',sources=["telplugins.i","_telplugins.cpp"]) 
  
setup(name=name, 
      version=version, 
      ext_modules=[ext_modules]) 