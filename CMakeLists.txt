CMAKE_MINIMUM_REQUIRED(VERSION 3.17.3)
set(CMAKE_CXX_STANDARD 20)
project(deeplearning)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    opencv_dir: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OPENCV_INCLUDE_DIRS} include)
add_subdirectory(src)