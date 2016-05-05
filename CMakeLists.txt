cmake_minimum_required(VERSION 3.3)
project(Multiple_View_Modeling)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB SOURCE_FILES
        "*.h"
        "*.cpp"
        "GCO/*.cpp"
        "GCO/*.h")

add_executable(Multiple_View_Modeling ${SOURCE_FILES})

find_package( OpenCV REQUIRED )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
