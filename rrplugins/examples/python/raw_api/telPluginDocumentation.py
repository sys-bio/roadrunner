import rrplugins
import os

#Create a plugin manager
pm = rrplugins.createPluginManager()

pluginName = "tel_add_noise" #"tel_levenberg_marquardt"
plugin = rrplugins.Plugin(pluginName)
if not plugin:
    print(rrplugins.getLastError())
    exit()

print(plugin.info())

print(plugin.listOfProperties())
print(plugin.listOfPropertyDescriptions())


#Get the manual for the plugin (bundled as embedded PDF)

plugin.viewManual()
