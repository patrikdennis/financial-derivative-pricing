cmake_minimum_required(VERSION 3.10)
project(AsianOptionPricerProject)

# Use C++11 (adjust as needed).
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Include directories so headers can be found.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)

# ------------------------------------------------------------------------------
# (Optional) OpenMP only required for HPC implementation:
# find_package(OpenMP REQUIRED)
# set(USE_OPENMP FALSE)
# if(OpenMP_CXX_FOUND)
#   set(USE_OPENMP TRUE)
# endif()

# ------------------------------------------------------------------------------

# NOTE: If speed_test.cpp or other tests reference HPC code,
#       you must include AsianOptionPricerHPC.cpp here.
set(ASIAN_LIB_SOURCES
    AsianOptionPricer.cpp
    AsianOptionPricerHPC.cpp
    utils/verification.cpp
)
add_library(AsianPricerLib ${ASIAN_LIB_SOURCES})

# openmp not used in current build
# if(USE_OPENMP)
#   target_link_libraries(AsianPricerLib PUBLIC OpenMP::OpenMP_CXX)
# endif()

# ------------------------------------------------------------------------------
# separate executables for each test file

# 1) price_comparison
add_executable(price_comparison
    test/price_comparison.cpp
)
target_link_libraries(price_comparison AsianPricerLib)
# if(USE_OPENMP)
#   target_link_libraries(price_comparison PUBLIC OpenMP::OpenMP_CXX)
# endif()

# 2) computation_time
add_executable(computation_time
    test/computation_time.cpp
)
target_link_libraries(computation_time AsianPricerLib)
# if(USE_OPENMP)
#   target_link_libraries(computation_time PUBLIC OpenMP::OpenMP_CXX)
# endif()

# 3) parity
add_executable(parity_test
    test/parity.cpp
)
target_link_libraries(parity_test AsianPricerLib)
# if(USE_OPENMP)
#   target_link_libraries(parity_test PUBLIC OpenMP::OpenMP_CXX)
# endif()

# 4) speed_test
add_executable(speed_test
    test/speed_test.cpp
)
target_link_libraries(speed_test AsianPricerLib)
# if(USE_OPENMP)
#   target_link_libraries(speed_test PUBLIC OpenMP::OpenMP_CXX)
# endif()

