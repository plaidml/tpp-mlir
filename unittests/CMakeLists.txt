set_target_properties(StandaloneUnitTests PROPERTIES FOLDER "Standalone Tests")

function(add_tpp_unittest name)
  add_unittest(StandaloneUnitTests ${name} ${ARGN})
endfunction()

add_tpp_unittest(TppUnitTests 
  Smoke.cpp
)

target_link_libraries(TppUnitTests
  PRIVATE
  TPPStandalone
  )
