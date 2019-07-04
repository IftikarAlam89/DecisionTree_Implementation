set(BOOST_ROOT /usr/local/Cellar/boost/1.69.0/)
set(BOOST_INCLUDE_DIR /usr/local/Cellar/boost/1.69.0/include/)
set(BOOST_LIBRARY_DIR /usr/local/Cellar/boost/1.69.0/lib/)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost COMPONENTS system filesystem timer REQUIRED)
find_package(Threads REQUIRED)

set(CLANG_DEFAULT_CXX_STDLIB "libc++")
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCES
        src/Bagging.cpp
        src/DataReader.cpp
        src/DecisionTree.cpp
        src/Question.cpp
        src/Leaf.cpp
        src/Node.cpp
        src/Calculations.cpp
        src/TreeTest.cpp src/main.cpp src/main.h)

set(HEADERS
        include/Bagging.hpp
        include/Dataset.hpp
        include/DataReader.hpp
        include/DecisionTree.hpp
        include/Question.hpp
        include/Leaf.hpp
        include/Node.hpp
        include/Utils.hpp
        include/Calculations.hpp
        include/TreeTest.hpp src/main.cpp src/main.h)
find_package(OpenMP REQUIRED)
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS} src/main.cpp src/main.h)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} Threads::Threads OpenMP::OpenMP_CXX)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Weffc++ -Wpedantic)
target_include_directories(${PROJECT_NAME} PUBLIC
        ${Boost_INCLUDE_DIR}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

if(Boost_FOUND)
    add_executable(dectree src/main.cpp)
    include_directories(${Boost_INCLUDE_DIR})
    target_link_libraries(dectree
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_SYSTEM_LIBRARY}
            ${PROJECT_NAME})
    target_include_directories(dectree PUBLIC ./ ${Boost_INCLUDE_DIR})
endif()

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Config.cmake
        DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${${PROJECT_NAME}_VERSION}
        COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION lib/cmake/${PROJECT_NAME})
