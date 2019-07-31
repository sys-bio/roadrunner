CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_PYTHON_EXAMPLES)

set(examples 	
#rrLevenbergMarquardtDemo.py
#testData.py
#lmFitTestModel.xml
)

install(    
FILES 
${examples} 
DESTINATION plugins/Examples
COMPONENT Python
)

