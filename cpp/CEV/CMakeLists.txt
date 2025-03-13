cmake_minimum_required(VERSION 3.10)
project(CEVPricerProject)

# Set the C++ standard and common compiler flags.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Add the current folder so headers can be found.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create a library from the CEV pricer source.
add_library(CEVPricerLib
    CEVPricer.cpp
)

# Create an executable for testing.
# The test file (test_parity.cpp) is located in the 'test' subfolder.
add_executable(cevp_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_parity.cpp
)

# Link the CEV pricer library to the test executable.
target_link_libraries(cevp_test CEVPricerLib)

