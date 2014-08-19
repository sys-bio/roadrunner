#include <stdlib.h>
#include <stdio.h>

__global__ void kern() {
    printf("kern\n");
}

void launchKern() {
    printf("launchKern\n");
    kern<<<1, 1>>>();
}