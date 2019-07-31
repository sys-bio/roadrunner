from teplugins import *

try:   
    #Create a plugin manager
    pm = createPluginManager()
    
    pluginName = "tel_test_model" 
    plugin = loadPlugin(pm, pluginName)  
        
    #Get the manual for the plugin (bundled as embedded PDF)
    ptr = getPluginManualAsPDF(plugin)
    nrOfBytes = getPluginManualNrOfBytes(plugin)
    manual = cast(ptr, POINTER(c_char * nrOfBytes))[0]
    
    #Check if there is a PDF available
    if nrOfBytes == 0:
        print 'This plugin does not have a manual.'
        exit()
    
    #create a (temporary) pdf file from the data
    outFName = pluginName + '.pdf'
    
    with open(outFName, 'wb') as output:
        output.write(manual)
        
    os.system('start ' + outFName) #a total guess
except Exception as e:
    print 'Problem: ' + `e`
        