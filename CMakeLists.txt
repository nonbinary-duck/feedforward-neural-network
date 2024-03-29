cmake_minimum_required(VERSION 3.12)

# Set the compiler flags
# Use Wall to give good feedback and use pthread so we can use threads
# We need POSIX Threads (Pthreads), a standard for threads, since C++ doesn't have a native implementation of it
# Wno-reorder disables the reorder warning (because it doesn't seem to affect this program)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-reorder")

# Set the project name
project(AIAssignmentOne)

# Give directories where header files are located
# Technically not needed as we are an executable application and main links to everything we need for us
include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

# Search for source files and add them
file(GLOB compiled_srcs
    # In src
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Add the main to the list of sources
list(APPEND compiled_srcs "main.cpp")

# Add the list of sources into an executable
add_executable(AIAssignmentOne ${compiled_srcs})

# Use c++ 20
set_property(TARGET AIAssignmentOne PROPERTY CXX_STANDARD 20)