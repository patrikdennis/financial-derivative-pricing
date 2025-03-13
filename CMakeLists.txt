cmake_minimum_required(VERSION 3.10)
project(OptionPricers)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the subdirectories (each module has its own CMakeLists.txt)
add_subdirectory(cpp/european)
add_subdirectory(cpp/asian)
add_subdirectory(cpp/CEV)

