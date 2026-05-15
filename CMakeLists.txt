cmake_minimum_required(VERSION 3.16)
project(NEOPFA_PLUS_MASTER)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(BUILD_FILES NeOPFA_PLUS/PTA.h NeOPFA_PLUS/PTA.cpp NeOPFA_PLUS/NeOPFA_PLUS_index.cpp)
add_executable(NeOPFA_PLUS_index ${BUILD_FILES})

set(ANN_FILES NeOPFA_PLUS/PTA.h NeOPFA_PLUS/PTA.cpp NeOPFA_PLUS/NeOPFA_PLUS_search_step01.cpp)
add_executable(NeOPFA_PLUS_search_step01 ${ANN_FILES})

set(ANN_FILES NeOPFA_PLUS/PTA.h NeOPFA_PLUS/PTA.cpp NeOPFA_PLUS/NeOPFA_PLUS_search_step02.cpp)
add_executable(NeOPFA_PLUS_search_step02 ${ANN_FILES})
