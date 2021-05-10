CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(TEL_PYTHON_EXAMPLES)

set(plugin_examples
)
install(    
FILES 
README.txt 
${plugin_examples}
DESTINATION examples/python/parameter_minimization/one_parameter	
COMPONENT examples
)
