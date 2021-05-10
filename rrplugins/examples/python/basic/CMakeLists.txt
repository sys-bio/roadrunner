CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(TEL_PYTHON_EXAMPLES)

set(plugin_examples
telCreateNoisyData.py
telMakeDataSeries.py
telPluginDocumentation.py
telReadDataFromFile.py
telWriteDataToFile.py
testData.dat

)
install(    
FILES 
README.txt 
${plugin_examples}
DESTINATION examples/python/basic 	
COMPONENT examples
)
