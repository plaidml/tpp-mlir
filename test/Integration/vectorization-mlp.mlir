// RUN: export vectorFile=$(mktemp)
// RUN: mlir-gen   --kernel=const --bias --relu --layers=16,16 --tiles=4,4,4 -float-type=bf16 --vnni=2  | tpp-run -linalg-to-vector -e entry --entry-point-result=void -print -n 1 --seed=123 2>&1 > $vectorFile
// RUN:  export xsmmFile=$(mktemp)
// RUN:  mlir-gen   --kernel=const --bias --relu --layers=16,16 --tiles=4,4,4 -float-type=bf16 --vnni=2  | tpp-run -e entry --entry-point-result=void  -print -n 1 --seed=123 2>&1 > $xsmmFile
// RUN: fpcmp -r 0.09 $vectorFile $xsmmFile
