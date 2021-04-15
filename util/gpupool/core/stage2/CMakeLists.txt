cmake_minimum_required(VERSION 3.14)
project(stage2)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_library(stage2 SHARED
        main.cpp
        ndarray.h)
