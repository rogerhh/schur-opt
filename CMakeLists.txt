cmake_minimum_required(VERSION 2.18)

project(schur-opt)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_executable(driver.exe driver.cpp schur_opt.h schur_opt.cpp)
