cmake_minimum_required(VERSION 3.10)

project(simple_cnn_tutorial)
add_subdirectory(uTensor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-rtti")

add_executable(simple_cnn_tutorial main.cpp models/MyModel.cpp)
target_include_directories(simple_cnn_tutorial PRIVATE models)

target_link_libraries(simple_cnn_tutorial utensor)
target_compile_features(simple_cnn_tutorial PUBLIC cxx_std_11)
