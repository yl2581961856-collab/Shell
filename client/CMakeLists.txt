cmake_minimum_required(VERSION 3.16)
project(asr_ws_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

add_executable(asr_ws_client
    main.cpp
)

target_include_directories(asr_ws_client PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(asr_ws_client PRIVATE
    Boost::system
    Threads::Threads
)
