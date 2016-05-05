cmake_minimum_required(VERSION 3.3)
project(Multiple_View_Modeling)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB sources
        "*.h"
        "*.cpp"
        "GCO/*.cpp"
        "GCO/*.h")

list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/GCO/example.cpp)
#list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/NormalOptimization.cpp)

add_executable(Multiple_View_Modeling ${sources})

find_package( OpenCV REQUIRED )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
