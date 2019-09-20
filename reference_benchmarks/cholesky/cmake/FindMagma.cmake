# Just force the user to specify where magma is installed on the command line for now.
find_package_handle_standard_args(Magma DEFAULT_MSG
  Magma_LIBRARIES
  Magma_LIB_DIR
  Magma_INCLUDE_DIR
)
