cmake_minimum_required(VERSION 3.13)
project(parla_runtime_libs C)

find_package(Python COMPONENTS Development)

set(CMAKE_C_STANDARD 11)

add_library(parla_supervisor SHARED virt_dlopen.c supervisor_wrappers.c log.c)
target_link_libraries(parla_supervisor dl)
install(TARGETS parla_supervisor DESTINATION "${CMAKE_INSTALL_LIBDIR}")

add_library(parla_context SHARED affinity_override.c log.c)
target_link_libraries(parla_context dl)
install(TARGETS parla_context DESTINATION "${CMAKE_INSTALL_LIBDIR}")

add_custom_target(python_stubs)

list(REMOVE_DUPLICATES Python_LIBRARIES)

function(forwarding_library lib)
    set(options VERSIONED)
    set(oneValueArgs OUTPUT_NAME SONAME)
    set(multiValueArgs)
    cmake_parse_arguments(forwarding_library
                          "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )
    get_filename_component(lib_full_name "${lib}" NAME)
    string(REGEX REPLACE "\\.so[.]*$" "" lib_libname "${lib_full_name}")
    string(REGEX REPLACE "lib" "" lib_name "${lib_libname}")
    string(REGEX REPLACE "lib" "" lib_name_with_ext "${lib_full_name}")
    add_custom_command(
        OUTPUT
            "${CMAKE_BINARY_DIR}/libstub_${lib_name_with_ext}.c"
#            "${CMAKE_BINARY_DIR}/libstub_${lib_name_with_ext}.versionscript"
        BYPRODUCTS "${CMAKE_BINARY_DIR}/libstub_${lib_name_with_ext}.versionscript.tmp"
        COMMAND bash "${CMAKE_SOURCE_DIR}/generate_stub_library.sh" "${lib}" "${forwarding_library_SONAME}"
        WORKING_DIR "${CMAKE_BINARY_DIR}"
        COMMENT "Generating stub library for ${lib_full_name}."
        VERBATIM
    )
    add_library("${lib_name}_parla_stub" SHARED "${CMAKE_BINARY_DIR}/libstub_${lib_name_with_ext}.c")
    install(TARGETS "${lib_name}_parla_stub" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    target_link_libraries("${lib_name}_parla_stub" dl)
    if(${forwarding_library_VERSIONED})
        target_link_options("${lib_name}_parla_stub" PRIVATE -Wl,--version-script=${CMAKE_BINARY_DIR}/libstub_${lib_name_with_ext}.versionscript)
    endif()
    target_include_directories("${lib_name}_parla_stub" PRIVATE ${CMAKE_SOURCE_DIR})
    set(${forwarding_library_OUTPUT_NAME} "${lib_name}" PARENT_SCOPE)
endfunction()

message("python libs: ${Python_LIBRARIES}")

foreach(pylib ${Python_LIBRARIES})
    message(${pylib})
    forwarding_library(${pylib} OUTPUT_NAME lib_name SONAME NULL)
    add_dependencies(python_stubs "${lib_name}_parla_stub")
endforeach(pylib)

find_package(CUDAToolkit)

forwarding_library(${CUDA_CUDART} VERSIONED)
forwarding_library(${CUDA_cuda_driver_LIBRARY})
