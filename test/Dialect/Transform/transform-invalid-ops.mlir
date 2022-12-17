// RUN: tpp-opt %s -verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{'transform.structured.pack' op attribute 'blocking_factors' failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  transform.structured.pack %arg0 { blocking_factors = [-1, 9] }
}
