cmake_minimum_required(VERSION 2.8)
PROJECT(auto_examples)

set(examples
bistable
)

foreach(example ${examples})
    add_subdirectory(${example})
#    FILE (GLOB hdrs ${example}/*.h)
#    FILE (GLOB src ${example}/*.cpp)
#
#    install (FILES ${hdrs}                          DESTINATION examples/${example}     COMPONENT example_files)
#    install (FILES ${src}                           DESTINATION examples/${example}     COMPONENT example_files)
#    install (FILES ${example}/Readme.txt            DESTINATION examples/${example}     COMPONENT example_files)
#    install (FILES ${example}/CMakeLists.txt        DESTINATION examples/${example}     COMPONENT example_files)
endforeach(example)
#
#install (FILES Readme.txt                       DESTINATION examples COMPONENT examples)
#install (FILES CMakeLists.txt                   DESTINATION examples COMPONENT examples)
