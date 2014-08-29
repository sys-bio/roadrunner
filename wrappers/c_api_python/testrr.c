#include <stdlib.h>
#include <stdio.h>

#include "rrc_api.h"

int main() {
    printf("main\n");
    RRHandle rr = createRRInstance();
    int result = loadSBMLFromFile(rr, "/home/jkm/devel/models/decayModel.xml");
    printf("  result = %d\n", result);
    printf("simulate\n");
    simulate(rr);
    return 0;
}