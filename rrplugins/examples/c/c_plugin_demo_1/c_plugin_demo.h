#ifndef c_plugin_demoH
#define c_plugin_demoH
#include "telCoreExporter.h"
//-----------------------------------------------------------
/*!
 \brief Global parameter holding last error, if any
*/
extern char*        gLastError;

//Global Plugin Data
TELHandle       gPlugin;
TELHandle       gDemoProperty;
TELHandle       gProperties;

//Each of these functions need to be present in order to load a plugin written in C
TLP_DS const char*     call_conv getImplementationLanguage(void);
TLP_DS const char*     call_conv getName(void);
TLP_DS const char*     call_conv getCategory(void);
TLP_DS bool            call_conv setupCPlugin(TELHandle aPlugin);
TLP_DS bool            call_conv execute(bool inThread);
TLP_DS bool            call_conv destroyPlugin(TELHandle plugin);
TLP_DS char*           call_conv getListOfCPluginPropertyNames(TELHandle plugin);
TLP_DS void*           call_conv getCPluginProperty(const char* name);

TLP_DS char*           call_conv getCLastError();
#endif



