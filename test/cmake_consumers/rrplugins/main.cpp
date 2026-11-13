#include <iostream>
#include "rrplugins/core/telPluginManager.h"
#include "rrplugins/common/telTelluriumData.h"
#include "telRandom.h"

int main() {
    tlp::TelluriumData data(2, 2);
    tlp::Random random(5);
    // The manager needs an existing folder to look for plugins in.
    tlp::PluginManager manager(".");
    std::cout << "rows " << data.rSize() << ", random " << (random.next() >= 0) << std::endl;
    return 0;
}
