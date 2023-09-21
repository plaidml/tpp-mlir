add_subdirectory(Xsmm)

if(USE_OneDNN)
  add_subdirectory(OneDnnl)
endif()
