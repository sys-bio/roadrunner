#include "GPUSimIntegratorInt.h"

#include <stdlib.h>
#include <stdio.h>

using namespace rr::rrgpu;

typedef float RKReal;

/**
 * @author JKM
 * @brief RK4 kernel
 * @param[in] n The size of the state vector
 */
__global__ void kern(int n) {
    extern __shared__ RKReal k[];
    printf("kern\n");
}

void launchKern(GPUSimIntegratorInt& intf) {
    printf("launchKern state vec size %d\n",  intf.getStateVectorSize());
    kern<<<1, 1, intf.getStateVectorSize()*4>>>(intf.getStateVectorSize());
    cudaDeviceSynchronize();
}
