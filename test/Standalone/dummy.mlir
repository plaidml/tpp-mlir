// RUN: standalone-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @foo()
    func @foo() {
        // CHECK: return
        return
    }
}
