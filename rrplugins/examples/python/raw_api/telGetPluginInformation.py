import rrplugins

# Create a plugin manager
pm = rrplugins.createPluginManager()
    
# Load plugins from the plugin folder
result = rrplugins.loadPlugins(pm)

print('Number of Plugins: ', rrplugins.getNumberOfPlugins(pm))
print('Plugin Names: ', rrplugins.getPluginNames(pm))

#Go trough all plugins and print(information
aPlugin = rrplugins.getFirstPlugin(pm)
while True:
    if aPlugin == None:
        break            
    print('==========================================')
    print('Name: ', rrplugins.getPluginName(aPlugin))
    print('Author: ', rrplugins.getPluginAuthor(aPlugin))
    print('Copyright: ', rrplugins.getPluginCopyright(aPlugin))
    print('Version: ', rrplugins.getPluginVersion(aPlugin))
    print('Category: ', rrplugins.getPluginCategory(aPlugin))
    print('Description:', rrplugins.getPluginDescription(aPlugin))
    print('Parameters: ', rrplugins.getListOfPluginPropertyNames(aPlugin))
    
    #get next plugin
    aPlugin = rrplugins.getNextPlugin(pm)

rrplugins.unLoadPlugins(pm)    
    
