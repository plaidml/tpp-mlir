set_target_properties(TPPUnitTests PROPERTIES FOLDER "TPP Tests")

function(add_tpp_unittest name)
  add_unittest(TPPUnitTests ${name} ${ARGN})
endfunction()

add_tpp_unittest(SmokeUnitTests 
  Smoke.cpp
)

target_link_libraries(SmokeUnitTests
  PRIVATE
  MLIRTPP
  )
