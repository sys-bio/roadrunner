#include <iostream>
#include <string>
#include "rrplugins/core/telPluginManager.h"

int main(int argc, char* argv[]) {
    // The folder to load plugins from: the one given on the command line, or the one the plugins
    // were installed to (RR_PLUGIN_FOLDER is set by CMakeLists.txt).
    std::string folder = argc > 1 ? argv[1] : RR_PLUGIN_FOLDER;

    // Creating the plugin manager loads all of the plugins in the folder.
    tlp::PluginManager manager(folder);

    std::cout << "Loaded " << manager.getNumberOfPlugins() << " plugins from " << folder << std::endl;
    tlp::StringList names = manager.getPluginNames();
    for (size_t i = 0; i < names.count(); ++i) {
        std::cout << "  " << names[i] << std::endl;
    }
    return 0;
}
