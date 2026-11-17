#include <iostream>
#include "rrplugins/core/telPluginManager.h"
#include "rrplugins/common/telTelluriumData.h"
#include "telRandom.h"

int main() {
    tlp::TelluriumData data(2, 2);
    tlp::Random random(5);
    // Creating the plugin manager loads all of the plugins in the folder.
    tlp::PluginManager manager(RR_PLUGIN_FOLDER);
    std::cout << "Loaded " << manager.getNumberOfPlugins() << " plugins" << std::endl;
    std::cout << "rows " << data.rSize() << ", random " << (random.next() >= 0) << std::endl;
    return 0;
}
