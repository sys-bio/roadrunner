#include "GPUSimIntegratorInt.h"

#include <stdlib.h>
#include <stdio.h>

using namespace rr::rrgpu;

__global__ void kern() {
    printf("kern\n");
}

void launchKern(GPUSimIntegratorInt& intf) {
    printf("launchKern\n");
    kern<<<1, 1>>>();
    cudaDeviceSynchronize();
}