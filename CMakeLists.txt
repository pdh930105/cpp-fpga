cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(
    c++-fpga
    VERSION 0.1
    DESCRIPTION "CPU-FPGA Heterogeneous System"
    )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}") #"${CMAKE_CXX_FLAGS}" command line option can be added

find_package(Torch REQUIRED)
find_package(OpenCL REQUIRED)
find_package(XRT REQUIRED)
find_package(OpenMP)

MESSAGE(STATUS "CMAKE CURRENT BINARY DIR ${CMAKE_CURRENT_BINARY_DIR}")
MESSAGE(STATUS "CMAKE CURRENT SOURCE DIR ${CMAKE_CURRENT_SOURCE_DIR}")
MESSAGE(STATUS "CMAKE SOURCE DIR ${CMAKE_SOURCE_DIR}")

if(NOT XRT_FOUND)
  MESSAGE(FATAL_ERROR "XRT not found, please source XRT setup script before building")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test/hw_emu_src/alveo_examples.xclbin")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernel/test/hw_emu_src/alveo_examples.xclbin ${CMAKE_CURRENT_BINARY_DIR}/alveo_examples.xclbin COPYONLY)
else()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/alveo_examples_u200.xclbin ${CMAKE_CURRENT_BINARY_DIR}/alveo_examples.xclbin COPYONLY)
endif()

# Library of utility functions common to all applications
add_library(example_utils STATIC 
  lib/xilinx_ocl_helper.cc 
  lib/event_timer.cc
)

target_include_directories(example_utils PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${XRT_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
)

# Basic kernel load/unload example
add_executable(main
  app/main.cc
  lib/event_timer.cc
  lib/xilinx_ocl_helper.cc
  )

target_include_directories(main PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${XRT_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
  ${TORCH_LIBRARIES}
  ${XRT_LIBS}
  ${OpenCL_LIBRARIES}
  pthread
  uuid
  ${CMAKE_DL_LIBS}
  example_utils
)
