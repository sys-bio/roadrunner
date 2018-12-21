cmake_minimum_required(VERSION 2.8)

PROJECT(rrModels)

set(models
README.txt
bistable.xml
feedback.xml
sbml_test_0001.xml 				
#ss_threestep.xml 	
#ss_threeSpecies.xml
)

if(BUILD_TESTS)
add_subdirectory(test_models)
endif()

install(    FILES ${models}
            DESTINATION models 	
            COMPONENT sbml_models)

install(    FILES README.txt
            DESTINATION models 	
            COMPONENT sbml_models)
