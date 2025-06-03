cmake_minimum_required(VERSION 3.10)
project(VideoStabilization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(video_stabilization 
    src/main.cpp
    src/main_utils.cpp
    src/stabilizer.cpp
    src/camera_engine.cpp
)

target_link_libraries(video_stabilization ${OpenCV_LIBS})
