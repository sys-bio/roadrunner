CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(TEL_PYTHON_EXAMPLES)


set(plugin_examples
telBifurcation.py
bistable.xml
BIOMD0000000203.xml
)

install(    
FILES 
README.txt 
${plugin_examples}
DESTINATION examples/python/bifurcation 	
COMPONENT examples
)
