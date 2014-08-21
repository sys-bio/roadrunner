#include "GPUSimIntegratorInt.h"

#include <stdlib.h>
#include <stdio.h>

using namespace rr::rrgpu;

typedef float RKReal;

#define RK4BLOCKS 4

// size in bytes of the RK coefficients with quad buffering etc.
#define RK_COEF_SIZE RK4BLOCKS*RK4BLOCKS*n*sizeof(RKReal)

#define RK_STATE_VEC_SIZE RK4BLOCKS*n*sizeof(RKReal)

#define RK_TIME_VEC_SIZE RK4BLOCKS*sizeof(RKReal)

/**
 * @author JKM
 * @brief RK4 kernel
 * @param[in] n The size of the state vector
 * @param[in] y The state vector
 */
__global__ void kern(int n,  RKReal h) {
    extern __shared__ RKReal k[];
    RKReal* f = k+RK_COEF_SIZE;
    RKReal* t = k+RK_COEF_SIZE+RK_STATE_VEC_SIZE;

//     RKReal t = t0;

    printf("kern\n");
    // hope you can render unicode
    // y ∈ ℝ(n)
    // so, to access k use pattern
    // offset = generation*RK4BLOCKS*n + block*n + i
    // where i is the index in ℝ(n)
    // and block corresponds to the index of the coefficient k1..4

    // initialize k
    for (int j=0; j<RK4BLOCKS; ++j)
        k[j*RK4BLOCKS*n + blockIdx.x*n + threadIdx.x] = 0;
    printf("k[%d*%d + %d] = %f\n", blockIdx.x, n, threadIdx.x, k[blockIdx.x*n + threadIdx.x]);

    // initialize state vector
    f[blockIdx.x*n + threadIdx.x] = 0;

    // initialize time vector
    t[0] = 0;
    t[1] = 0.5*h; // 0.5f?
    t[2] = 0.5*h;
    t[3] = h;

    // current generation
    int m=0;

    while (m < RK4BLOCKS) {
//         k[m*RK4BLOCKS*n + blockIdx.x*n + threadIdx.x] = f[blockIdx.x*n + threadIdx.x];
        ++m;
    }

}

void launchKern(GPUSimIntegratorInt& intf) {
    int n = intf.getStateVectorSize();

    printf("launchKern state vec size %d\n", n);

    RKReal* y;
    cudaMalloc(&y, n*sizeof(RKReal));

    printf("launchKern RK_COEF_SIZE %d\n", RK_COEF_SIZE, RK_STATE_VEC_SIZE, RK_TIME_VEC_SIZE);

    // execution configuration
    // * first param: num blocks (four blocks for RK4)
    // * second param: threads per block
    // * third param: shared memory size
    // shared mem for k's: RK4BLOCKS*RK4BLOCKS*n*sizeof(float)
    // shared mem for state vectors: RK4BLOCKS*n*sizeof(float)
    // set to size of the state vector*4 because it is
    // quadruple buffered (enables concurrent execution)
//     kern<<<RK4BLOCKS, n,
//         RK_COEF_SIZE + // RK coefficients
//         RK_STATE_VEC_SIZE + // state vector
//         RK_TIME_VEC_SIZE // time vector
//         >>>(n, 0.1);

    cudaDeviceSynchronize();

    cudaFree(y);
}
