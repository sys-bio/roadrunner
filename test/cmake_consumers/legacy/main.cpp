#include "rr/rrVersionInfo.h"
#include <iostream>

int main() {
    std::cout << rr::getVersionStr() << std::endl;
    return 0;
}
