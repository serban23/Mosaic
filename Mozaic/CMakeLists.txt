cmake_minimum_required(VERSION 3.30)
project(Mozaic)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Mozaic
        main.cpp
        src/mozaic.cpp
        src/mozaic.h
)

target_link_libraries(Mozaic ${OpenCV_LIBS})