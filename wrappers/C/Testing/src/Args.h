#ifndef CommandLineParametersH
#define CommandLineParametersH
#include <string>
using std::string;

string Usage(const string& prg);
class Args
{
    public:
                                        Args();
        virtual                        ~Args(){}
        string                          ModelsFilePath;     //option m:
        string                          ResultOutputFile;   //option r:
        string                          TempDataFolder;     //option t:
        bool                            EnableLogging;      //option v:
        string                          Suites;             //option s:
        string                          compiler;           //option c:
        string                          TestDataFolder;     //option i:
};

#endif
