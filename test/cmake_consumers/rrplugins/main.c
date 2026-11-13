#include <stdio.h>
#include "telplugins_c_api.h"

int main(void) {
    printf("plugins version %s\n", tpGetVersion());
    return 0;
}
