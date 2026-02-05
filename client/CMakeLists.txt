cmake_minimum_required(VERSION 3.16)
project(asr_ws_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

option(ASRWS_ENABLE_PORTAUDIO "Enable microphone streaming via PortAudio" OFF)

add_executable(asr_ws_client
    main.cpp
)

target_include_directories(asr_ws_client PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(asr_ws_client PRIVATE
    Boost::system
    Threads::Threads
)

if(ASRWS_ENABLE_PORTAUDIO)
    find_path(PORTAUDIO_INCLUDE_DIR NAMES portaudio.h)
    find_library(PORTAUDIO_LIBRARY NAMES portaudio portaudio_static portaudio64)

    if(NOT PORTAUDIO_INCLUDE_DIR OR NOT PORTAUDIO_LIBRARY)
        message(FATAL_ERROR "PortAudio not found. Install it (e.g. vcpkg install portaudio) or build with -DASRWS_ENABLE_PORTAUDIO=OFF.")
    endif()

    target_compile_definitions(asr_ws_client PRIVATE ASRWS_ENABLE_PORTAUDIO=1)
    target_include_directories(asr_ws_client PRIVATE ${PORTAUDIO_INCLUDE_DIR})
    target_link_libraries(asr_ws_client PRIVATE ${PORTAUDIO_LIBRARY})
endif()
