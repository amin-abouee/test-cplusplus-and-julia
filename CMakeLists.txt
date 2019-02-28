cmake_minimum_required(VERSION 3.5)
project(Test_Eigen)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++14 -fdiagnostics-color=always")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

find_package(Threads)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/unsupported/)
add_executable(eigen_test main.cpp)
