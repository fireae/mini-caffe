project(MiniCaffe-Examples)
cmake_minimum_required(VERSION 2.8)

set(OpenCV_DIR "d:/Apps/opencv/build")
find_package(OpenCV REQUIRED)
include(../mini-caffe.cmake)

add_executable(deeplandmark deeplandmark/landmark.hpp deeplandmark/landmark.cpp deeplandmark/main.cpp)
target_link_libraries(deeplandmark caffe ${OpenCV_LIBS})

add_executable(wgan wgan/main.cpp)
target_link_libraries(wgan caffe ${OpenCV_LIBS})

add_executable(r-fcn r-fcn/main.cpp)
target_link_libraries(r-fcn caffe ${OpenCV_LIBS})

add_executable(ssh ssh/main.cpp)
target_link_libraries(ssh caffe ${OpenCV_LIBS})

add_executable(ocr ocr/main.cpp)
target_link_libraries(ocr caffe ${OpenCV_LIBS})

add_executable(classification classification/main.cpp)
target_link_libraries(classification caffe ${OpenCV_LIBS})

add_executable(ssd ssd/main.cpp)
target_link_libraries(ssd caffe ${OpenCV_LIBS})
