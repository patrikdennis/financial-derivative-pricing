cmake_minimum_required(VERSION 3.10)
project(EuropeanPricer)

# Ensure the current directory is added so that headers are found
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# List source files 
set(SOURCES
    EuropeanPricer.cpp
    # tests need to be added and will be
    test_european.cpp
)

# Create an executable
add_executable(cpp_pricer_european ${SOURCES})

