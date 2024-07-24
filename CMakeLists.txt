cmake_minimum_required(VERSION 3.12)

project(mission2)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mission2 main.cpp)

target_link_libraries(mission2 ${OpenCV_LIBS})

