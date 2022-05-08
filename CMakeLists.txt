cmake_minimum_required(VERSION 3.11)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(schur-opt)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_executable(driver.exe driver.cpp schur_opt.h schur_opt.cpp)
add_executable(driver_debug.exe driver_gpu.cu schur_opt_gpu.h schur_opt_debug.cpp)

# Packaging system
set(CPACK_GENERATOR TGZ)
install(FILES gpu.cu DESTINATION .)
include(CPack)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(driver_gpu.exe driver_gpu.cu schur_opt_gpu.h schur_opt_gpu.cu)
target_link_libraries(driver_gpu.exe cublas.so)
target_compile_features(driver_gpu.exe PRIVATE cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

configure_file(job-schur_opt_gpu job-schur_opt_gpu COPYONLY)
