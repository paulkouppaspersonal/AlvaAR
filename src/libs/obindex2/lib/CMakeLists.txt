cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(obindex2 LANGUAGES CXX)

# Compile tests or not
option(EnableTesting "EnableTesting" ON)

# Setting build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Setting the flags for profiling information or not
if(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Setting Release options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Setting Debug options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1")
endif()

# C++ standard is set via CMAKE_CXX_STANDARD from parent build
set(CMAKE_CXX_STANDARD 17)
message(STATUS "Using C++17 standard.")

# Packages required to work with the library
find_package(OpenCV REQUIRED) # OpenCV

# Printing the compiling flags
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

include_directories(include
                    external
                    ${OpenCV_INCLUDE_DIRS}
                    )

link_directories(${OpenCV_LIBRARIES}
                 )

### Targets ###
add_library(obindex2
            src/binary_descriptor.cc
            src/binary_tree_node.cc
            include/obindex2/priority_queues.h
            src/binary_tree.cc
            src/binary_index.cc)
target_link_libraries(obindex2
                      ${OpenCV_LIBRARIES}
                      )

set_target_properties(obindex2 PROPERTIES LINK_FLAGS "${CMAKE_CXX_FLAGS}")

install(TARGETS obindex2 DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
