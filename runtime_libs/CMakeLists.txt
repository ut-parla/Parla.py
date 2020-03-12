cmake_minimum_required(VERSION 3.12)
project(parla_runtime_libs C)

find_package(Python COMPONENTS Development)

set(CMAKE_C_STANDARD 11)

add_library(parla_supervisor SHARED virt_dlopen.c supervisor_wrappers.c log.c)
target_link_libraries(parla_supervisor dl)

add_library(parla_context SHARED affinity_override.c log.c)
target_link_libraries(parla_context dl)

add_custom_target(python_stubs)

list(REMOVE_DUPLICATES Python_LIBRARIES)

message("python libs: ${Python_LIBRARIES}")

foreach(pylib ${Python_LIBRARIES})
    message(${pylib})
    get_filename_component(pylib_full_name "${pylib}" NAME)
    string(REGEX REPLACE "\\.so[.]*$" "" pylib_libname "${pylib_full_name}")
    string(REGEX REPLACE "lib" "" pylib_name "${pylib_libname}")
    string(REGEX REPLACE "lib" "" pylib_name_with_ext "${pylib_full_name}")
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/libstub_${pylib_name_with_ext}.c"
        COMMAND bash "${CMAKE_SOURCE_DIR}/generate_stub_library.sh" "${pylib}"
        WORKING_DIR "${CMAKE_BINARY_DIR}"
        COMMENT "Generating stub library for ${pylib_full_name}."
        VERBATIM
    )
        
    #add_custom_command(
    #    "generate_${pylib_name}_stub"
    #    COMMAND "bash ${CMAKE_SOURCE_DIR}/generate_stub_library.sh ${pylib}"
    #    OUTPUT "${CMAKE_BINARY_DIR}/libstub_${pylib_name}.c"
    #    WORKING_DIR "${CMAKE_BINARY_DIR}"
    #    COMMENT "Generating stub library for ${pylib}"
    #    VERBATIM
    #)
    add_library("${pylib_name}_stub" SHARED "${CMAKE_BINARY_DIR}/libstub_${pylib_name_with_ext}.c")
    #add_dependencies("${pylib_name}_stub" "generate_${pylib_name}_stub")
    add_dependencies(python_stubs "${pylib_name}_stub")
endforeach(pylib)
