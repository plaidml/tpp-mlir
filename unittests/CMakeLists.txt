set_target_properties(StandaloneUnitTests PROPERTIES FOLDER "Standalone Tests")

function(add_tpp_unittest name)
  add_unittest(StandaloneUnitTests ${name} ${ARGN})
endfunction()

add_tpp_unittest(TppUnitTests 
  smoke.cpp 
  )

target_link_libraries(TppUnitTests
  PRIVATE
  MLIRStandalone
  )
