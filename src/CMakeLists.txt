add_executable(${PROJECT_NAME} main.cpp window.cpp detector.cpp transform.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})