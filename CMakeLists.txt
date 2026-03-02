cmake_minimum_required (VERSION 3.0...4.1.2)
project(zk-pof)
set(NAME "zk-pof")

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/emp-base.cmake)

find_package(emp-tool REQUIRED)
find_package(emp-zk REQUIRED)
#find_package(emp-ag2pc REQUIRED)

IF (${EMP-ZK_FOUND})
    MESSAGE(STATUS "Found ZK.")
ENDIF (${EMP-ZK_FOUND})
include_directories(${EMP-TOOL_INCLUDE_DIRS})
include_directories(${EMP-ZK_INCLUDE_DIRS})
#include_directories(${EMP-AG2PC_INCLUDE_DIRS})

ENABLE_TESTING()
ADD_SUBDIRECTORY(test)