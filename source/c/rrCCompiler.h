#ifndef rrCompilerH
#define rrCompilerH
#include <vector>
#include <string>
#include "rrCompiler.h"
#include "rrStringList.h"

using std::vector;
using std::string;

namespace rr
{

/**
 * compiler class for the C based model system.
 */
class RR_DECLSPEC CCompiler : public Compiler
{
public:
    CCompiler(const string& supportCodeFolder, const string& compiler);
    virtual ~CCompiler();

    virtual string              getCompiler() const;
    virtual bool                setCompiler(const string& compiler);

    bool                        setupCompiler(const string& supportCodeFolder);
    bool                        compile(const string& cmdLine);

    virtual bool                setCompilerLocation(const string& path);
    virtual string              getCompilerLocation() const;

    virtual bool                setSupportCodeFolder(const string& path);
    virtual string              getSupportCodeFolder() const;

    bool                        setIncludePath(const string& path);
    bool                        setLibraryPath(const string& path);
    void                        execute(StringList& oProxyCode);
    bool                        compileSource(const string& cSource);
    string                      getCompilerMessages();
    bool                        setOutputPath(const string& path);


    virtual std::string getDefaultTargetTriple();

    virtual std::string getProcessTriple();

    virtual std::string getHostCPUName();

    virtual std::string getVersion();

private:
    string                      mDLLFileName;
    string                      mSupportCodeFolder;
    string                      mCompilerName;
    string                      mCompilerLocation;    //Path to executable

    vector<string>              mCompilerOutput;
    vector<string>              mIncludePaths;
    vector<string>              mLibraryPaths;
    vector<string>              mCompilerFlags;
    string                      createCompilerCommand(const string& sourceFileName);
    bool                        setupCompilerEnvironment();
    string                      mOutputPath;

};

} //namespace rr


#endif
