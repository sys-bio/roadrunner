from teplugins import *

try:    
    # Create a plugin manager
    pm = createPluginManager()
        
    # Load plugins from the plugin folder
    result = loadPlugins(pm)
    
    print 'Number of Plugins: ' + `getNumberOfPlugins(pm)`
    print 'Plugin Names: ' + `getPluginNames(pm)`
    
    #Go trough all plugins and print information
    aPlugin = getFirstPlugin(pm)
    while True:
        if aPlugin == None:
            break            
        print '=========================================='
        print 'Name: ' + `getPluginName(aPlugin)`
        print 'Author: ' + `getPluginAuthor(aPlugin)`
        print 'Copyright: ' + `getPluginCopyright(aPlugin)`
        print 'Version: ' + `getPluginVersion(aPlugin)`
        print 'Category: ' + `getPluginCategory(aPlugin)`                        
        print 'Description:' + `getPluginDescription(aPlugin)`
        print 'Parameters: ' + `getListOfPluginPropertyNames(aPlugin)`                    
        
        #get next plugin
        aPlugin = getNextPlugin(pm)
    
    unLoadPlugins(pm)    
    
except Exception as e:
    print 'There was a problerm: ' + `e`    
