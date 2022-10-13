// RUN: standalone-opt %s -verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects blocking factors to be positive integers, found [-1, 9]}}
  transform.structured.pack %arg0 { blocking_factors = [-1, 9] }
}
