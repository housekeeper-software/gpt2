# 最低 CMake 版本要求
cmake_minimum_required(VERSION 3.20)

# 项目名称和版本
project(gpt-2 VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(PROJECT_SRC_ROOT ${CMAKE_SOURCE_DIR}/src)

add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS -D_HAS_EXCEPTIONS=0 -DUNICODE -D_UNICODE)


set(PCRE2_USE_STATIC_LIBS ON)
find_package(PCRE2 CONFIG COMPONENTS 8BIT)

message(STATUS "PCRE2_FOUND: ${PCRE2_FOUND}")
message(STATUS "PCRE2_INCLUDE_DIRS: ${PCRE2_INCLUDE_DIR}")
message(STATUS "PCRE2_LIBRARIES: ${PCRE2_LIBRARIES}")


set(DENSE_TENSOR "${THIRD_PARTY_PATH}/tensor/sources")

set(dense_tensor
    ${DENSE_TENSOR}/safe_tensors.cc
    ${DENSE_TENSOR}/storage.cc
    ${DENSE_TENSOR}/tensor.cc
    ${DENSE_TENSOR}/tensor_string.cc
)

# 添加你的源代码文件
# 将你的 .cc 文件列在这里
file(GLOB_RECURSE SRC ${PROJECT_SRC_ROOT}/*.c*)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SRC} ${dense_tensor})

# 设置运行时输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

# 设置调试时的工作目录，CMake Tools 会读取此属性
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})


target_compile_definitions(${PROJECT_NAME} PRIVATE PCRE2_STATIC)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PCRE2_LIBRARIES})

# 添加头文件路径
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SRC_ROOT}
    ${THIRD_PARTY_PATH}/json
    ${PCRE2_INCLUDE_DIR}
    ${DENSE_TENSOR}
)
