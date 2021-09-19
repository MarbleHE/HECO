// RUN: abc-opt %s | abc-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = abc.foo %{{.*}} : i32
        %res = abc.foo %0 : i32
        return
    }
}
