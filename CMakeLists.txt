cmake_minimum_required(VERSION 2.8)

project(demo)
add_compile_options(-std=c++11)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

find_package( G2O REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Cholmod )
include_directories(${EIGEN3_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
    ${CHOLMOD_LIBRARIES})
