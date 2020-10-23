set(target ExamplesCAPITester)

add_executable( ${target} main.cpp)

target_link_libraries (${target} PRIVATE roadrunner_c_api)
target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES})

install(
        TARGETS ${target} 		
        DESTINATION bin 
        COMPONENT compiled_examples)
